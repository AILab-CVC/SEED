from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler

# from utils.logging import AverageMeter

from transformers import CLIPTokenizer
from utils.data_utils import filter_no_caption_or_no_image, filter_no_cls_or_no_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import torch
import numpy as np
import pyarrow.parquet as pq
import io
import math
import webdataset as wds
import json
import braceexpand
import ast

from torch.utils.data import Dataset

from PIL import Image
import PIL
import logging
import os

from data.policies import CenterCropSDTransform
from torch.utils.data import default_collate

import utils.distributed as dist_utils
import pandas as pd

from utils.logging import Path

from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)


class ParquetDataset:

    def __init__(
        self,
        path,
        tokenizer,
        batch_size=256,
        workers=1,
        train=True,
        num_processes=1,
        rank=None,
        **kwargs,
    ):
        super(ParquetDataset, self).__init__()

        use_cuda = torch.cuda.is_available()

        # Data loading code
        loading_kwargs = ({"num_workers": workers, "pin_memory": True} if use_cuda else {})

        # Data loading code
        self.dataset = self.get_dataset(path, tokenizer=tokenizer, train=train, **kwargs)
        self.batch_size = batch_size

        if num_processes > 1:
            print(f"[{rank}] initializing DistributedSampler")
            shuffle = None
            self.sampler = DistributedSampler(self.dataset, num_replicas=num_processes, rank=rank, shuffle=train)
        else:
            shuffle = train
            self.sampler = None

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            **loading_kwargs,
        )

    def get_dataset(self, root, tokenizer, **kwargs):
        dataset = ParquetTextImageDataset(
            data_root=root,
            tokenizer=tokenizer,
            **kwargs,
        )

        return dataset


class WebDataset(object):

    def __init__(
        self,
        path,
        tokenizer,
        num_examples_to_see,
        batch_size=256,
        workers=1,
        train=True,
        resolution=512,
        filters=None,
        **kwargs,
    ):
        self.filters = filters or {}
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.workers = workers
        self.dataset = self.get_dataset(
            path,
            tokenizer=tokenizer,
            train=train,
            num_examples_to_see=num_examples_to_see,
            filters=self.filters,
        )

        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,  # Shuffling done in the webdataset
            num_workers=workers,
            persistent_workers=True,
        )

        logging.info(f"Unused dataset parameters for WebDataset: {kwargs}")

    def get_dataset(self, url, tokenizer, train, num_examples_to_see, filters):
        transform = CenterCropSDTransform(center_crop=True, size=self.resolution)

        pipeline = [wds.ResampledShards(url)]

        # TODO: Currently does not support validation sampling well
        # Don't split by worker and node since we're sampling with replacement
        # if train:
        #     pipeline.append(wds.shuffle(2000))

        pipeline.extend([
            tarfile_to_samples_nothrow,
        ])

        if train:
            pipeline.append(wds.shuffle(2000))

        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.select(metadata_filters(filters)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(pixel_values="jpg;png;jpeg;webp", input_ids="txt", text_raw="txt"),
            wds.map(filter_keys(set(["pixel_values", "input_ids", "text_raw"]))),
            wds.map_dict(
                pixel_values=transform,
                input_ids=lambda text: tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0],
                text_raw=lambda text: text,
            ),
            wds.batched(self.batch_size, partial=not train, collation_fn=default_collate),
        ])

        effective_batch_size = dist_utils.compute_effective_batch_size(self.batch_size)

        num_worker_batches = math.ceil(num_examples_to_see / (effective_batch_size * self.workers))

        # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size

        return wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)


class ClassificationWebDataset(object):

    def __init__(
        self,
        path,
        tokenizer,
        num_examples_to_see,
        class_mapping,
        batch_size=256,
        workers=1,
        train=True,
        resolution=512,
        **kwargs,
    ):
        if isinstance(class_mapping, dict):
            self.class_mapping = class_mapping
        elif isinstance(class_mapping, os.PathLike) or isinstance(class_mapping, str):
            self.class_mapping = json.load(Path(class_mapping).open("r"))
        else:
            raise TypeError(f"{type(class_mapping)} not accepted, need str or os.PathLike")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.resolution = resolution
        self.workers = workers

        self.dataset = self.get_dataset(
            path,
            tokenizer=tokenizer,
            train=train,
            num_examples_to_see=num_examples_to_see,
        )

        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,  # Shuffling done in the webdataset
            num_workers=workers,
            persistent_workers=True,
        )

        logging.info(f"Unused dataset parameters for WebDataset: {kwargs}")

    def get_dataset(self, url, tokenizer, train, num_examples_to_see):
        transform = CenterCropSDTransform(center_crop=True, size=self.resolution)

        pipeline = [wds.ResampledShards(url)]

        if train:
            pipeline.append(wds.shuffle(100))

        pipeline.extend([
            tarfile_to_samples_nothrow,
        ])

        if train:
            pipeline.append(wds.shuffle(1000))

        pipeline.extend([
            wds.select(filter_no_cls_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(
                pixel_values="jpg;png;jpeg;webp",
                input_ids="cls",
                text_raw="cls",
                class_idx="cls",
            ),
            wds.map(filter_keys(set(["pixel_values", "input_ids", "text_raw", "class_idx"]))),
            wds.map_dict(
                pixel_values=transform,
                input_ids=lambda class_idx: tokenizer(
                    self.class_mapping[str(class_idx)],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0],
                text_raw=lambda class_idx: self.class_mapping[str(class_idx)],
            ),
            wds.batched(self.batch_size, partial=not train, collation_fn=default_collate),
        ])

        effective_batch_size = dist_utils.compute_effective_batch_size(self.batch_size)

        num_worker_batches = math.ceil(num_examples_to_see / (effective_batch_size * self.workers))

        # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size

        return wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)


class CSVDataset(object):

    def __init__(
        self,
        path,
        tokenizer,
        num_examples_to_see,
        batch_size=256,
        workers=1,
        train=True,
        resolution=512,
        filters=None,
        sample_with_replacement=False,
        **kwargs,
    ):
        self.filters = filters or {}
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.workers = workers

        self.dataset = self.get_dataset(
            path,
            tokenizer=tokenizer,
            train=train,
            num_examples_to_see=num_examples_to_see,
            filters=self.filters,
        )

        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,  # Shuffling done in the webdataset
            num_workers=workers,
            persistent_workers=True,
        )

        logging.info(f"Unused dataset parameters for CSV Dataset: {kwargs}")

    def get_dataset(self, csv_path, tokenizer, train, num_examples_to_see, filters):
        transform = CenterCropSDTransform(center_crop=True, size=self.resolution)

        self.csv = pd.read_csv(csv_path)


class TorchCSVDataset(Dataset):

    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        sep="\t",
        tokenizer=None,
        filter_keys=None,
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.iter_df = self.filter_dataset(df, filter_keys)

        self.img_key = img_key
        self.caption_key = caption_key
        self.transforms = transforms

        logging.debug("Done loading data.")
        self.tokenize = tokenizer or (lambda x: x)

    def filter_dataset(self, df, filter_keys):
        if filter_keys:
            filter_fn = metadata_filters(filter_keys)
        else:
            filter_fn = None

        out_dict = defaultdict(list)

        logging.debug("Processing CSV dataframe files")
        num_filtered = 0
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            select = filter_fn(row_dict)

            if select:
                for k, v in row_dict.items():
                    out_dict[k].append(v)
            else:
                num_filtered += 1

        logging.debug(f"Filtered {num_filtered} out of {len(df)}")

        return out_dict

    def __len__(self):
        return len(self.iter_df)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]

        return images, texts


def metadata_filters(filter_dict):

    def filter_fn(sample):
        select = True
        if "json" not in sample:
            # No 'json' in sample to use for filtering
            # - if `filter_dict`` is not empty, then we should not select this sample
            # - if `filter_dict`` is empty, it means there is no filter and thus
            # we select the sample
            return False if filter_dict else True

        db = json.loads(sample["json"])

        for param, expr in filter_dict.items():
            if param not in db:
                logging.info("Field {param} not in sample")
                return False

            param_val = db[param]

            # TODO: This allows code injection
            select = select and eval(f"{param_val}{expr}")

            # if ">" in val:
            #     threshold = float(val.split(">")[-1])
            #     select = select and (param_val > threshold)
            # elif "<" in val:
            #     threshold = float(val.split("<")[-1])
            #     select = select and (param_val < threshold)
            # else:
            #     raise ValueError("Need direction for filter threshold")

        if not select:
            logging.info(f"Field {param} not match threshold")

        return select

    return filter_fn


def filter_keys(key_set):

    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
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
        if (current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample):
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


class ParquetTextImageDataset(Dataset):

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        center_crop=False,
        train=False,
        flip_p=0.0,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=flip_p)
        self.db = self._process_parquet(self.data_root)

        self.num_images = len(self.db)
        self._length = self.num_images

    def __len__(self):
        return self._length

    def _process_parquet(self, path):
        db = pq.read_table(path).to_pandas()

        return db

    def __getitem__(self, i):
        example = {}

        # Image processing
        image = Image.open(io.BytesIO(self.db.image[i]["bytes"]))
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        # Text processing
        text = self.db.text[i]

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["text"] = text

        return example


def test_kwargs():
    return {
        "data_root": "/home/ramanv/pokemon.parquet",
        "tokenizer": CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer"),
    }


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")

    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])

    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, "r").read())

    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258

    num_shards = len(shards_list)

    return total_size, num_shards


def dummy_test(**kwargs):
    model_id = "CompVis/stable-diffusion-v1-4"
    url = "/usr/data/chunks_merged_1e3/shard_{00000..00008}.tar"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    default_kwargs = {"batch_size": 64, "num_examples_to_see": 128, "workers": 4}
    default_kwargs.update(**kwargs)

    return WebDataset(path=url, tokenizer=tokenizer, train=True, **default_kwargs)


def dummy_pipe_test():
    ds = WebDataset(
        path="pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar -",
        tokenizer=CLIPTokenizer.from_pretrained("/fsx/home-vkramanuj/stable-diffusion-v1-5", subfolder="tokenizer"),
        num_examples_to_see=100,
    )

    return ds


def classification_wds_test():
    tokenizer = CLIPTokenizer.from_pretrained("/fsx/home-vkramanuj/stable-diffusion-v1-5", subfolder="tokenizer")

    ds = ClassificationWebDataset(
        path="pipe:aws s3 cp s3://s-laion/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar -",
        tokenizer=tokenizer,
        num_examples_to_see=100000,
        class_mapping="scripts/metadata/imagenet_idx_to_prompt.json",
        workers=5,
    )

    return ds