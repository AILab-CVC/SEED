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

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


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


class detshuffle2(wds.PipelineStage):

    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


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


def get_caption_dataset(
    shards_url_or_list,
    num_samples,
    image_processor=None,
    tokenizer=None,
    epoch=0,
    resample=True,
    batch_size=4,
    num_workers=1,
    max_length=128,
    floor=False,
    seed=0,
):
    if isinstance(shards_url_or_list, str):
        shards_url_or_list = [shards_url_or_list]

    if not isinstance(num_samples, list):
        num_samples = [num_samples] * len(shards_url_or_list)

    assert len(shards_url_or_list) == len(num_samples)

    total_shards = []
    total_num_samples = 0

    for shards_url, num_sample in zip(shards_url_or_list, num_samples):
        shards = braceexpand.braceexpand(shards_url)
        total_shards.extend(shards)
        total_num_samples += num_sample

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_text_fn = functools.partial(preprocess_text, tokenizer=tokenizer, max_length=max_length)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    if resample:
        pipeline = [ResampledShards2(shards_url, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(shards_url)]

    # at this point we have an iterator over all the shards
    if not resample:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=seed,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ])
    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        # wds.tarfile_to_samples(handler=log_and_continue),
        tarfile_to_samples_nothrow,
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
    ])