import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import pickle
import random
import os
from braceexpand import braceexpand
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torch.utils.data.dataloader import default_collate
import base64
import io
import tarfile
from torchdata.datapipes.iter import TarArchiveLoader
from typing import cast, IO, Iterable, Iterator, Optional, Tuple, Dict
from torchdata.datapipes import functional_datapipe
from io import BufferedIOBase
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
import warnings
from torchdata.datapipes.iter import IterDataPipe
import hydra
from omegaconf import OmegaConf
import pyrootutils

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


@functional_datapipe("load_from_tar_wo_exception")
class TarArchiveLoaderWoException(TarArchiveLoader):
    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(data_stream.file_obj, tarfile.TarFile):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (self.mode if hasattr(data_stream, "seekable") and data_stream.seekable() else
                                    self.mode.replace(":", "|"))
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(fileobj=cast(
                        Optional[IO[bytes]], data_stream), mode=reading_mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(
                            f"failed to extract file {tarinfo.name} from source tarfile {pathname}")
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(
                        os.path.join(pathname, tarinfo.name))
                    # type: ignore[misc]
                    yield inner_pathname, StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!")
                # raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()


def decode_obelisc_data_for_llm(item, tokenizer=None, max_length=512, reverse_ratio=0.5, max_images=None):
    key, value = item

    sample = {}
    if key.endswith(".pkl"):
        try:
            value = pickle.load(value)
        except Exception as e:
            print(f'Error occured when load pkl: {e}')
            return key, sample

        image_list = value['image_ids']
        text_list = value['texts']
        metadata = value['metadata']

        reverse_flag = np.random.uniform(0, 1) < reverse_ratio
        if reverse_flag:
            idx = 0
            while idx < len(image_list) - 1:
                if image_list[idx] is not None:
                    image_list[idx], image_list[idx +
                                                1] = image_list[idx + 1], image_list[idx]
                    text_list[idx], text_list[idx +
                                              1] = text_list[idx + 1], text_list[idx]
                    idx += 2
                else:
                    idx += 1

        unified_tokens = tokenizer.bos_token
        cur_images = 0
        for image_id, text in zip(image_list, text_list):
            if (image_id is None) + (text is None) != 1:
                print('Incorrect data format, skip.')
                return key, {}
            if image_id is not None:
                if max_images is not None and cur_images >= max_images:
                    break
                image_tokens = BOI_TOKEN + \
                    ''.join([IMG_TOKEN.format(int(item))
                            for item in image_id]) + EOI_TOKEN
                unified_tokens += image_tokens
                cur_images += 1
            else:
                unified_tokens += text
                if max_images is not None and cur_images >= max_images:
                    break
        unified_tokens += tokenizer.eos_token

        tokenized = tokenizer(unified_tokens,
                              max_length=max_length,
                              add_special_tokens=False,
                              truncation=True,
                              padding='max_length',
                              return_tensors='pt')

        input_ids = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]
        labels = torch.clone(input_ids)
        labels[labels == tokenizer.pad_token_id] = -100
        filter_flag = True

        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'filter_flag': filter_flag,
        }
    else:
        return key, None


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    if 'metadata' not in unwarpped:
        unwarpped['metadata'] = '{}'
    if '__key__' in unwarpped:
        unwarpped['__key__'] = unwarpped['__key__'].split('/')[-1]
    return unwarpped


def filter_data_for_llm(item):
    if 'input_ids' in item and item.get('filter_flag', True):
        return True
    else:
        print('A sample has been filtered out.')
        return False


def build_obelisc_datapipes_for_llm(data_dir,
                                    tokenizer=None,
                                    max_length=512,
                                    reverse_ratio=0.5,
                                    max_images=None,
                                    recursive=True,
                                    batch_size=None,
                                    cycle_count=None):
    """
    datapipe of image interleaved dataset (such as mmc4...) with webdataset format
    """

    decode_partial = functools.partial(decode_obelisc_data_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       reverse_ratio=reverse_ratio,
                                       max_images=max_images)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    # datapipe = dp.iter.FileLister(root=data_dir, masks='mmc4-0-000004.tar', recursive=True)
    datapipe = dp.iter.FileLister(
        root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


if __name__ == '__main__':
    from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
    data_dir = './obelisc'
    seed_tokenizer_cfg_path = ''
    dataloader_num_workers = 2

    seed_tokenizer_cfg = OmegaConf.load(seed_tokenizer_cfg_path)
    seed_tokenizer = hydra.utils.instantiate(seed_tokenizer_cfg)

    dataset = build_obelisc_datapipes_for_llm(
        data_dir=data_dir, tokenizer=seed_tokenizer, max_length=1024, cycle_count=1)

    mp_rs = MultiProcessingReadingService(
        num_workers=dataloader_num_workers)
    dist_rs = DistributedReadingService()
    rs = SequentialReadingService(dist_rs, mp_rs)

    dataloader = DataLoader2(dataset, reading_service=rs)
