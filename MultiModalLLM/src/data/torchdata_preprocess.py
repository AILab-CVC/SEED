import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import os
import pickle
import random
import base64
import io
import torch
from torch.utils.data.dataloader import default_collate
from datasets import load_dataset

from braceexpand import braceexpand


def decode_image_text_pair(item,
                           tokenizer=None,
                           image_processor=None,
                           image_transform=None,
                           max_length=128,
                           use_caption_in_metadata=False,
                           caption_key_in_metadata=''):
    key, value = item
    if key.endswith(".txt"):
        if not use_caption_in_metadata:
            caption = value.read().decode('utf-8')
            if tokenizer is None:
                return key, {'text': caption}
            else:
                tokenized = tokenizer(
                    caption,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return key, {
                    'input_ids': tokenized['input_ids'][0],
                    'attention_mask': tokenized['attention_mask'][0],
                    'text': caption
                }
        else:
            return key, {}
    elif key.endswith(".jpg"):
        try:
            image = Image.open(value).convert('RGB')
            width, height = image.size
        except Exception as e:
            print('Error while decode image: ', e)
            return key, None

        if image_processor is not None:
            try:
                image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]
            except Exception as e:
                print('Error while process image: ', e)
                return key, None
        elif image_transform is not None:
            try:
                image_tensor = image_transform(image)
            except Exception as e:
                print('Error while transform image: ', e)
                return key, None
        else:
            image_tensor = image
        return key, {'pixel_values': image_tensor, 'height': height, 'width': width}
    elif key.endswith(".json"):
        metadata_str = value.read().decode('utf-8')
        if use_caption_in_metadata:

            metadata = json.loads(metadata_str)
            caption = metadata[caption_key_in_metadata]
            if tokenizer is None:
                return key, {'text': caption, 'metadata': metadata_str}
            else:
                tokenized = tokenizer(
                    caption,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return key, {
                    'input_ids': tokenized['input_ids'][0],
                    'attention_mask': tokenized['attention_mask'][0],
                    'text': caption,
                    'metadata': metadata_str
                }
        else:
            return key, {'metadata': metadata_str}
    else:
        return key, {}


def decode_text_in_metadata(item, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    key, value = item
    if key.endswith(".txt"):
        caption = value.read().decode('utf-8')
        return key, {
            'original_text': caption,
        }

    elif key.endswith(".jpg"):
        try:
            image = Image.open(value).convert('RGB')
        except Exception as e:
            print('Error while decode image: ', e)
            return key, None

        if image_processor is not None:
            try:
                image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]
            except Exception as e:
                print('Error while process image: ', e)
                return key, None
        elif image_transform is not None:
            try:
                image_tensor = image_transform(image)
            except Exception as e:
                print('Error while transform image: ', e)
                return key, None
        else:
            image_tensor = image
        return key, {'pixel_values': image_tensor}
    elif key.endswith(".json"):
        metadata = value.read().decode('utf-8')
        metadata_dict = json.loads(metadata)
        caption = metadata_dict['top_caption']
        image_name = str(metadata_dict['key']) + str(metadata_dict['hash'])
        if tokenizer is None:
            return key, {'text': caption, 'metadata': metadata, 'image_name': image_name}
        else:
            tokenized = tokenizer(
                caption,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return key, {
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'text': caption,
                'metadata': metadata,
                'image_name': image_name,
            }
    else:
        return key, value


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


def filter_data(item):
    if 'pixel_values' in item and 'text' in item:
        return True
    else:
        print('filtered')
        return False


def filter_data_like_qwen(item, similarity_thr=0.2, min_resolution=180, min_aspect_ratio=0.666):
    if ('pixel_values' not in item) or ('text' not in item) or (not item.get('filter_flag', True)):
        print('filtered because filter flag.')
        return False
    else:
        metadata = json.loads(item['metadata'])

        if 'width' in item:
            width = item['width']
        elif 'width' in metadata:
            width = metadata['width']
        elif 'WIDTH' in metadata:
            width = metadata['WIDTH']
        elif 'resolution' in metadata:
            width = metadata['resolution'][0]
        elif 'original_resolution' in metadata:
            width = metadata['original_resolution'][0]
        elif 'original_width' in metadata:
            width = metadata['original_width']
        else:
            width = None

        if 'height' in item:
            height = item['height']
        elif 'height' in metadata:
            height = metadata['height']
        elif 'HEIGHT' in metadata:
            height = metadata['HEIGHT']
        elif 'resolution' in metadata:
            height = metadata['resolution'][1]
        elif 'original_resolution' in metadata:
            height = metadata['original_resolution'][1]
        elif 'original_height' in metadata:
            height = metadata['original_height']
        else:
            height = None

        if 'all_similarities' in metadata:
            similarity = max(metadata['all_similarities'])
        elif 'similarity' in metadata:
            similarity = metadata['similarity']
        elif 'score' in metadata:
            similarity = metadata['score']
        elif 'SCORE' in metadata:
            similarity = metadata['SCORE']
        else:
            similarity = None

        if height is not None and width is not None:

            aspect_ratio = height / width
            if height < min_resolution or width < min_resolution:
                # print(f'filtered because resolution: ({width},{height})')
                return False
            if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
                # print(f'filtered because aspect ratio: ({width},{height})')
                return False
        if similarity is not None:
            if similarity < similarity_thr:
                return False

        return True


def build_caption_datapipes_with_pixels(data_dir,
                                        tokenizer=None,
                                        image_processor=None,
                                        image_transform=None,
                                        max_length=128,
                                        batch_size=None,
                                        similarity_thr=0.2,
                                        min_resolution=180,
                                        min_aspect_ratio=0.666,
                                        use_caption_in_metadata=False,
                                        caption_key_in_metadata='top_caption'):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    # decode_partial = functools.partial(decode_text_in_metadata,
    decode_partial = functools.partial(decode_image_text_pair,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length,
                                       use_caption_in_metadata=use_caption_in_metadata,
                                       caption_key_in_metadata=caption_key_in_metadata)

    filter_partial = functools.partial(filter_data_like_qwen,
                                       similarity_thr=similarity_thr,
                                       min_resolution=min_resolution,
                                       min_aspect_ratio=min_aspect_ratio)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    # datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    # datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_partial)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
    return datapipe


def build_multi_caption_datapipes(data_dirs,
                                  concat_type='concat',
                                  tokenizer=None,
                                  image_processor=None,
                                  image_transform=None,
                                  max_length=128):
    assert concat_type in ['concat', 'mux']

    datasets = []
    for data_dir in data_dirs:
        dataset = build_caption_datapipes_with_pixels(data_dir=data_dir,
                                                      tokenizer=tokenizer,
                                                      image_processor=image_processor,
                                                      image_transform=image_transform,
                                                      max_length=max_length)
        datasets.append(dataset)

    if concat_type == 'concat':
        datapipe = dp.iter.Concater(*datasets)
    elif concat_type == 'mux':
        datapipe = dp.iter.MultiplexerLongest(*datasets)
    else:
        raise NotImplementedError

    return datapipe


# def get_sample_karpathy_list(data):
#     file_name, json_data = data
#     for item in json_data:
#         print(item)
#         if 'caption' in item and 'image' in item:
#             yield item


def decode_coco_image_text_pair(item, image_root, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    caption = item['caption']
    image_name = item['image']
    image_path = os.path.join(image_root, image_name)
    image = Image.open(image_path).convert('RGB')

    metadata = {'image_id': item['image_id']}
    if 'id' in item:
        metadata['id'] = item['id']

    sample = {'metadata': json.dumps(metadata), 'text': caption}

    if tokenizer is not None:
        tokenized = tokenizer(
            caption,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample['input_ids'] = tokenized['input_ids'][0]
        sample['attention_mask'] = tokenized['attention_mask'][0]

    if image_processor is not None:
        try:
            image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]
            sample['pixel_values'] = image_tensor
        except Exception as e:
            print('Error while process image: ', e)
    elif image_transform is not None:
        try:
            image_tensor = image_transform(image)
            sample['pixel_values'] = image_tensor
        except Exception as e:
            print('Error while transform image: ', e)
    else:
        sample['pixel_values'] = image

    return sample


def build_coco_caption_datapipes(data_dir,
                                 mask=None,
                                 recursive=False,
                                 tokenizer=None,
                                 image_processor=None,
                                 image_transform=None,
                                 max_length=128,
                                 batch_size=None):

    decode_partial = functools.partial(decode_coco_image_text_pair,
                                       image_root=data_dir,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)

    mask = '*.json' if mask is None else mask
    datapipe = dp.iter.FileLister(root=data_dir, masks=mask, recursive=recursive)
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_json_files()
    datapipe = datapipe.parse_coco_annotations()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)

    datapipe = datapipe.filter(filter_data)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
    return datapipe


def base64_to_image(base64_str: str) -> Image.Image:
    img_data = base64.b64decode(base64_str)
    img_buffer = io.BytesIO(img_data)
    img = Image.open(img_buffer).convert('RGB')
    return img


def decode_mmc4_data(item, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    key, value = item

    sample = {}
    if key.endswith(".pkl"):
        try:
            value = pickle.load(value)
        except Exception as e:
            print(f'Error occured when load pkl: {e}')
            return key, sample

        image_list = value['images']
        text_list = value['texts']
        metadata = value['metadata']
        # image_flag = [image is not None for image in image_list]
        # image_list = [base64_to_image(image) for image in image_list if image is not None]

        image_flag = []
        new_text_list = []
        new_image_list = []
        for image_base64, text in zip(image_list, text_list):
            assert (image_base64 is None) + (text is None) == 1
            if image_base64 is not None:
                try:
                    image = base64_to_image(image_base64)
                    new_image_list.append(image)
                    new_text_list.append(text)
                    image_flag.append(True)
                except Exception as e:
                    print(f'Error occured when convert image_base64: {e}')
            else:
                new_text_list.append(text)
                image_flag.append(False)

        sample['texts'] = json.dumps(new_text_list)
        sample['image_flag'] = json.dumps(image_flag)
        sample['metadata'] = json.dumps(metadata)

        image_list = new_image_list
        if len(image_list) == 0 and (image_transform is not None or image_processor is not None):
            sample['pixel_values'] = torch.tensor([])

        elif image_transform is not None:
            image_tensors = [image_transform(image) for image in image_list]
            image_tensors = torch.stack(image_tensors, dim=0)
            sample['pixel_values'] = image_tensors
        elif image_processor is not None:
            image_tensors = [image_processor(image, return_tensors='pt')['pixel_values'][0] for image in image_list]
            image_tensors = torch.stack(image_tensors, dim=0)
            sample['pixel_values'] = image_tensors
        else:
            sample['pixel_values'] = image_list

        return key, sample


def collate_mmc4_data(batch):
    pixel_values = [sample.pop('pixel_values') for sample in batch]
    batch = default_collate(batch)
    if isinstance(pixel_values[0], torch.Tensor):
        batch['pixel_values'] = torch.concat(pixel_values, dim=0)
    else:
        batch['pixel_values'] = pixel_values

    return batch


def filter_with_pixels(item):
    if 'pixel_values' in item:
        return True
    else:
        print('filtered')
        return False


def build_mmc4_datapipes_with_pixels(data_dir,
                                     tokenizer=None,
                                     image_processor=None,
                                     image_transform=None,
                                     max_length=128,
                                     batch_size=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_mmc4_data,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    # datapipe = dp.iter.FileLister(root=data_dir, masks='mmc4-0-000004.tar', recursive=True)
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_with_pixels)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(collate_fn=collate_mmc4_data)
    return datapipe


def decode_ra_data_low_level(item):
    key, value = item

    if key.endswith(".pkl"):
        new_value = {}
        value = pickle.load(value)
        new_value['text'] = value['text']
        new_value['image_name'] = str(value['metadata']['key']) + str(value['metadata']['hash'])
        new_context = []
        for item in value['context']:
            new_context.append({
                'text': item['text'],
                'image_name': item['image_name'],
                'similarity': item['similarity'],
            })
        new_value['context'] = new_context
        new_value['metadata'] = value['metadata']

        return key, {'data': json.dumps(new_value)}
    else:
        return key, None


def build_ra_update_datapipes(data_dir):

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_ra_data_low_level)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)

    return datapipe


def decode_m3it_data_with_pixel_values(item, image_transform=None):
    instruction = item["instruction"]  # str
    inputs = item["inputs"]  # str
    outputs = item["outputs"]  # str
    image_base64_str_list = item["image_base64_str"]  # str (base64)

    image_list = []
    image_flag = []
    for image_base64 in image_base64_str_list:
        if image_base64 is not None:
            try:
                image = base64_to_image(image_base64)
                image_list.append(image)
                image_flag.append(True)
            except Exception as e:
                print(f'Error occured when convert image_base64: {e}')

    sample = {}
    sample['instruction'] = instruction
    sample['inputs'] = inputs
    sample['outputs'] = outputs
    sample['image_flag'] = json.dumps(image_flag)

    if len(image_list) == 0 and (image_transform is not None):
        sample['pixel_values'] = torch.tensor([])

    elif image_transform is not None:
        image_tensors = [image_transform(image) for image in image_list]
        image_tensors = torch.stack(image_tensors, dim=0)
        sample['pixel_values'] = image_tensors
    else:
        sample['pixel_values'] = image_list

    return sample


def build_m3it_datapipes_with_pixel_values(data_file, ds_name_list, cache_dir, image_transform=None, batch_size=None):
    datapipes = []
    for ds_name in ds_name_list:
        dataset = load_dataset(data_file, ds_name, cache_dir=cache_dir)
        if dataset.get('train', None) is not None:
            datapipes.append(dp.iter.IterableWrapper(dataset['train']))
        if dataset.get('validation', None) is not None:
            datapipes.append(dp.iter.IterableWrapper(dataset['validation']))

    decode_partial = functools.partial(decode_m3it_data_with_pixel_values, image_transform=image_transform)

    datapipe = dp.iter.Concater(*datapipes)
    # datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(collate_fn=collate_mmc4_data)
    return datapipe


def decode_llava_data_with_pixel_values(item, image_dir, image_prefix='', image_transform=None):
    key, value = item
    image_name = value['image']
    sample = {}
    image_path = os.path.join(image_dir, image_prefix + image_name)
    image_list = []
    image_flag = []
    filter_flag = True
    try:
        image = Image.open(image_path).convert('RGB')
        image_list.append(image)
        image_flag.append(True)

        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 3 or aspect_ratio < 1 / 3:
            filter_flag = False

    except Exception as e:
        print(f'Error occured when loading image: {e}, skip')
        return sample

    sample['data'] = json.dumps(value['data'])
    sample['image_flag'] = json.dumps(image_flag)
    sample['filter_flag'] = filter_flag
    if len(image_list) == 0 and (image_transform is not None):
        sample['pixel_values'] = torch.tensor([])

    elif image_transform is not None:
        image_tensors = [image_transform(image) for image in image_list]
        image_tensors = torch.stack(image_tensors, dim=0)
        sample['pixel_values'] = image_tensors
    else:
        sample['pixel_values'] = image_list

    return sample


def filter_with_pixels_and_flag(item):
    if 'pixel_values' in item and item.get('filter_flag', True):
        return True
    else:
        print('filtered')
        return False


def build_llava_datapipes_with_pixel_values(ann_dir,
                                            ann_mask,
                                            image_dir,
                                            image_prefix='',
                                            image_transform=None,
                                            batch_size=None):
    if isinstance(ann_dir, str):
        ann_dir = list(braceexpand(ann_dir))

    decode_partial = functools.partial(decode_llava_data_with_pixel_values,
                                       image_dir=image_dir,
                                       image_prefix=image_prefix,
                                       image_transform=image_transform)

    datapipe = dp.iter.FileLister(root=ann_dir, masks=ann_mask, recursive=True)
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_with_pixels_and_flag)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(collate_fn=collate_mmc4_data)

    return datapipe
