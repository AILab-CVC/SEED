import ast
import functools
import io
import json
import logging
import math
import os
import random
import sys
import tarfile
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import torch
import torch.utils
import torchvision
import webdataset as wds
from PIL import Image, ImageSequence
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

# Image.MAX_IMAGE_PIXELS = 1000000000
# MAX_NUM_TOKENS = 256
# MAX_NUM_IMAGES = 5
# TINY_IMAGE_SIZE_THRESHOLD = 1
# N_CHANNELS = 3
# INTERLEAVED_IMAGE_SIZE = 224


class SharedEpoch:

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def filter_no_caption_or_no_image(sample):
    return ("txt" in sample) and ("png" in sample or "jpg" in sample or "jpeg" in sample)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(exn):  # Avoid spamming logs with these
        return True
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def preprocess_image(sample, image_processor):
    image = [image_processor.preprocess(s, return_tensors="pt")["pixel_values"] for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    # image = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(image)
    return image


def preprocess_text(sample, tokenizer, max_length=128):
    text = tokenizer(
        sample,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return text['input_ids'], text["attention_mask"]


# def preprocess_text(sample, tokenizer, max_length=128):
#     return sample

# def split_shard_by_rank(shard_urls, rank, world_size):
#     if isinstance(shard_urls, str):
#         shard_list = list(braceexpand.braceexpand(shard_urls))
#     elif isinstance(shard_urls, list):
#         shard_list = shard_urls
#     else:
#         raise TypeError(f"Wrong type for shard_urls: {type(shard_urls)}")

#     step = math.ceil(len(shard_list) / world_size)
#     start = step * rank
#     end = min(start + step, len(shard_list))
#     return shard_list[start:end]


def get_caption_dataset(
    shards_url,
    image_processor=None,
    tokenizer=None,
    batch_size=4,
    num_workers=1,
    max_length=128,
):

    # shards_list = split_shard_by_rank(shard_urls, rank, world_size)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=0)
    pipeline = [wds.SimpleShardList(shards_url)]

    # at this point we have an iterator over all the shards

    pipeline.extend([
        wds.split_by_node,
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
    ])

    # at this point, we have an iterator over the shards assigned to each worker at each node

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_text_fn = functools.partial(preprocess_text, tokenizer=tokenizer, max_length=max_length)

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
        wds.batched(batch_size, partial=True),
        wds.map_tuple(preprocess_image_fn, preprocess_text_fn, handler=log_and_continue),
    ])

    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)
