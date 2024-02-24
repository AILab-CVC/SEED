import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import pickle
import random
from braceexpand import braceexpand
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torch.utils.data.dataloader import default_collate
import base64
import io

import hydra
from omegaconf import OmegaConf
import pyrootutils

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


def decode_edit_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None):
    key, value = item

    if caption_prompt is None:
        caption_prompt = '{}'
    if isinstance(caption_prompt, str):
        caption_prompt = [caption_prompt]

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        if 'source_image_ids' in sample.keys():
            source_image_ids = sample['source_image_ids']
            target_image_ids = sample['target_image_ids']

            source_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in source_image_ids]) + EOI_TOKEN
            target_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in target_image_ids]) + EOI_TOKEN

            instruction = sample['instruction'].replace('Edit this image with the instruction: ', '')
            unified_tokens = tokenizer.bos_token + source_image_tokens + instruction + target_image_tokens + tokenizer.eos_token
            num_ids = 34

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

            tokenized_question = tokenizer(instruction,
                                           max_length=max_length,
                                           add_special_tokens=False,
                                           truncation=True,
                                           padding="longest",
                                           return_tensors='pt')
            attention_mask_question = tokenized_question['attention_mask'][0]
            num_question_ids = attention_mask_question.sum()

            labels[:1 + num_ids + num_question_ids] = -100

        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'filter_flag': True,
        }
    else:
        return key, None


def decode_image_text_pair_for_llm(item,
                                   tokenizer=None,
                                   max_length=128,
                                   caption_prompt=None,
                                   reverse_ratio=0.5,
                                   mask_left_label=False,
                                   use_caption_in_metadata=False,
                                   caption_key_in_metadata=''):
    key, value = item

    if caption_prompt is None:
        caption_prompt = '{}'
    if isinstance(caption_prompt, str):
        caption_prompt = [caption_prompt]

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        image_ids = sample['image_ids']

        if not use_caption_in_metadata:
            text = sample['text']
        else:
            # print(type(sample['metadata']), sample['metadata'])
            text = sample['metadata'][caption_key_in_metadata]

        prompt = random.choice(caption_prompt)
        caption = prompt.format(text)

        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN

        reverse_flag = np.random.uniform(0, 1) < reverse_ratio
        if reverse_flag:
            unified_tokens = tokenizer.bos_token + image_tokens + caption + tokenizer.eos_token
        else:
            unified_tokens = tokenizer.bos_token + caption + image_tokens + tokenizer.eos_token

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
        # print('labels before: ',labels)

        filter_flag = True
        if text.strip(' ;,[]{}\'\".?:') == '':
            filter_flag = False

        if mask_left_label:
            try:
                if reverse_flag:
                    eoi_token_id = tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0]
                    eoi_idx = torch.where(labels == eoi_token_id)[0][0]
                    labels[:eoi_idx + 1] = -100
                else:
                    boi_token_id = tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0]
                    boi_idx = torch.where(labels == boi_token_id)[0][0]
                    labels[:boi_idx + 1] = -100
            except Exception as e:
                print('Error occured when masking label: ', e)
                filter_flag = False

        # print('labels after: ',labels)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text,
            'filter_flag': filter_flag,
        }
    else:
        return key, None


def decode_image_text_pair_with_pixels(item, tokenizer, image_processor=None, image_transform=None, max_length=128):
    key, value = item
    if key.endswith(".txt"):
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
        # return key, {'metadata': value.read().decode('utf-8')}
        return key, None
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


def filter_data_for_llm(item):
    if 'input_ids' in item and item.get('filter_flag', True):
        return True
    else:
        print('A sample has been filtered out.')
        return False


def filter_data_with_image_text(item):
    # print(item.keys())
    if 'pixel_values' in item and 'input_ids' in item:
        return True
    else:
        print('A sample has been filtered out.')
        return False


def build_caption_datapipes_for_llm(data_dir,
                                    tokenizer=None,
                                    max_length=128,
                                    caption_prompt=None,
                                    reverse_ratio=0.5,
                                    mask_left_label=False,
                                    use_caption_in_metadata=False,
                                    caption_key_in_metadata='top_caption',
                                    recursive=True,
                                    batch_size=None,
                                    cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(
        decode_image_text_pair_for_llm,
        tokenizer=tokenizer,
        max_length=max_length,
        caption_prompt=caption_prompt,
        reverse_ratio=reverse_ratio,
        mask_left_label=mask_left_label,
        use_caption_in_metadata=use_caption_in_metadata,
        caption_key_in_metadata=caption_key_in_metadata,
    )
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
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


def build_edit_datapipes_for_llm(data_dir, tokenizer=None, max_length=128, recursive=True, batch_size=None, cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(
        decode_edit_pair_for_llm,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
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


def build_multi_caption_datapipes_for_llm(data_dirs,
                                          concat_type='sample',
                                          sample_weights=None,
                                          tokenizer=None,
                                          max_length=128,
                                          caption_prompt=None,
                                          reverse_ratio=0.5,
                                          mask_left_label=False,
                                          recursive=True,
                                          batch_size=None,
                                          cycle_counts=None):
    assert concat_type in ['concat', 'mux_longest', 'sample']

    if sample_weights is None:
        sample_weights = [1] * len(data_dirs)
    else:
        assert len(sample_weights) == len(data_dirs)

    if cycle_counts is None or isinstance(cycle_counts, (int, float)):
        cycle_counts = [cycle_counts] * len(data_dirs)

    datasets_to_weights_dict = {}
    datasets = []
    for data_dir, sample_weight, cycle_count in zip(data_dirs, sample_weights, cycle_counts):
        dataset = build_caption_datapipes_for_llm(data_dir=data_dir,
                                                  tokenizer=tokenizer,
                                                  max_length=max_length,
                                                  caption_prompt=caption_prompt,
                                                  reverse_ratio=reverse_ratio,
                                                  mask_left_label=mask_left_label,
                                                  recursive=recursive,
                                                  batch_size=batch_size,
                                                  cycle_count=cycle_count)
        datasets_to_weights_dict[dataset] = sample_weight
        datasets.append(dataset)

    if concat_type == 'concat':
        datapipe = dp.iter.Concater(*datasets)
    elif concat_type == 'mux_longest':
        datapipe = dp.iter.MultiplexerLongest(*datasets)
    elif concat_type == 'sample':
        datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict)
    else:
        raise NotImplementedError

    return datapipe


def build_multi_datapipes(datapipes, tokenizer=None, concat_type='sample', sample_weights=None):
    assert concat_type in ['concat', 'mux_longest', 'sample']
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [hydra.utils.instantiate(datapipe, tokenizer=tokenizer) for datapipe in datapipes]

    if concat_type == 'concat':
        datapipe = dp.iter.Concater(*datapipes)
    elif concat_type == 'mux_longest':
        datapipe = dp.iter.MultiplexerLongest(*datapipes)
    elif concat_type == 'sample':
        datasets_to_weights_dict = {}
        for dataset, sample_weight in zip(datapipes, sample_weights):
            datasets_to_weights_dict[dataset] = sample_weight
        datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict)

    else:
        raise NotImplementedError

    return datapipe


def build_caption_datapipes_with_pixels(data_dir,
                                        masks=None,
                                        tokenizer=None,
                                        image_processor=None,
                                        image_transform=None,
                                        max_length=128,
                                        batch_size=None):

    decode_partial = functools.partial(decode_image_text_pair_with_pixels,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)
    masks = '*.tar' if masks is None else masks
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks=masks, recursive=True)
    datapipe = datapipe.cycle()
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_with_image_text)
    datapipe = datapipe.shuffle(buffer_size=512)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_multi_caption_datapipes_with_pixels(data_dirs,
                                              masks=None,
                                              concat_type='sample',
                                              sample_weights=None,
                                              tokenizer=None,
                                              image_processor=None,
                                              image_transform=None,
                                              max_length=128):
    assert concat_type in ['concat', 'mux_longest', 'sample']

    if sample_weights is None:
        sample_weights = [1] * len(data_dirs)
    else:
        assert len(sample_weights) == len(data_dirs)

    if masks is None:
        masks = ['*.tar'] * len(data_dirs)
    else:
        assert len(masks) == len(data_dirs)

    datasets_to_weights_dict = {}
    datasets = []
    for data_dir, mask, sample_weight in zip(data_dirs, masks, sample_weights):
        dataset = build_caption_datapipes_with_pixels(data_dir=data_dir,
                                                      masks=mask,
                                                      tokenizer=tokenizer,
                                                      image_processor=image_processor,
                                                      image_transform=image_transform,
                                                      max_length=max_length)
        datasets_to_weights_dict[dataset] = sample_weight
        datasets.append(dataset)

    if concat_type == 'concat':
        datapipe = dp.iter.Concater(*datasets)
    elif concat_type == 'mux_longest':
        datapipe = dp.iter.MultiplexerLongest(*datasets)
    elif concat_type == 'sample':
        datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict)
    else:
        raise NotImplementedError

    return datapipe


def base64_to_image(base64_str: str) -> Image.Image:
    img_data = base64.b64decode(base64_str)
    img_buffer = io.BytesIO(img_data)
    img = Image.open(img_buffer).convert('RGB')
    return img


def decode_mmc4_data_for_llm(item, tokenizer=None, max_length=512, reverse_ratio=0.5, max_images=None):
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
                    image_list[idx], image_list[idx + 1] = image_list[idx + 1], image_list[idx]
                    text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
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
                image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_id]) + EOI_TOKEN
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


def filter_with_pixels(item):
    if 'pixel_values' in item:
        return True
    else:
        print('filtered')
        return False


def build_mmc4_datapipes_for_llm(data_dir,
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

    decode_partial = functools.partial(decode_mmc4_data_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       reverse_ratio=reverse_ratio,
                                       max_images=max_images)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    # datapipe = dp.iter.FileLister(root=data_dir, masks='mmc4-0-000004.tar', recursive=True)
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
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


def convert_list_to_input_ids_and_attention_mask(data, tokenizer, mask_left_label=False, reverse_flag=False, max_length=512):
    input_ids = []
    labels = []

    for item in data:
        image_ids = item['image_ids']
        text = item['text']
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN

        if reverse_flag:
            left_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
            right_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            right_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
            left_ids = tokenizer.encode(text, add_special_tokens=False)

        input_ids_item = left_ids + right_ids
        if mask_left_label:
            labels_item = [-100] * len(left_ids) + right_ids
        else:
            labels_item = input_ids_item

        input_ids.extend(input_ids_item)
        labels.extend(labels_item)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def decode_ra_data_for_llm(item,
                           tokenizer=None,
                           max_length=512,
                           reverse_ratio=0.5,
                           max_images=None,
                           random_num_images=False,
                           min_images=None,
                           mask_left_label=False):
    key, value = item

    sample = {}
    if key.endswith(".pkl"):
        try:
            value = pickle.load(value)
        except Exception as e:
            print(f'Error occured when load pkl: {e}')
            return key, sample

        image_ids = value['image_ids']
        text = value['text']
        context = value['context']
        filter_flag = True

        if random_num_images:
            assert min_images is not None and max_images is not None
            num_images = random.randint(min_images, max_images)
        elif max_images is not None:
            num_images = max_images
        else:
            num_images = len(context) + 1

        if num_images > 1:
            random.shuffle(context)
            context = context[:num_images - 1]
            context = [{'text': text, 'image_ids': image_ids}] + context
        else:
            context = [{'text': text, 'image_ids': image_ids}]
        reverse_flag = np.random.uniform(0, 1) < reverse_ratio

        tokenized = convert_list_to_input_ids_and_attention_mask(
            data=context,
            tokenizer=tokenizer,
            mask_left_label=mask_left_label,
            reverse_flag=reverse_flag,
            max_length=max_length,
        )

        for i in range(len(context)):
            context[i]['image_ids'] = context[i]['image_ids'].tolist()

        return key, {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['labels'],
            'text': json.dumps(context),
            'filter_flag': filter_flag,
        }
    else:
        return key, None


def build_retrieval_augmented_datapipes_for_llm(data_dir,
                                                tokenizer=None,
                                                max_length=512,
                                                reverse_ratio=0.5,
                                                max_images=None,
                                                random_num_images=False,
                                                min_images=None,
                                                mask_left_label=True,
                                                recursive=True,
                                                batch_size=None,
                                                cycle_count=None):
    """
    datapipe of image interleaved dataset (such as mmc4...) with webdataset format
    """

    decode_partial = functools.partial(decode_ra_data_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       reverse_ratio=reverse_ratio,
                                       max_images=max_images,
                                       random_num_images=random_num_images,
                                       min_images=min_images,
                                       mask_left_label=mask_left_label)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle(buffer_size=50000)
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


def decode_coco_cap_eval_data(item, tokenizer=None, caption_prompt=None, num_context=2):
    key, value = item
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        image_ids = sample['image_ids']
        context = ''
        if num_context != 0:
            for item in sample['context'][:num_context]:
                context += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(image_id)) for image_id in item['image_ids']]) + EOI_TOKEN
                context += item['text']

        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN
        unified_tokens = tokenizer.bos_token + context + image_tokens
        if caption_prompt is not None:
            unified_tokens = unified_tokens + caption_prompt

        tokenized = tokenizer(unified_tokens, add_special_tokens=False, return_tensors='pt')
        return key, {
            'input_ids': tokenized['input_ids'][0],
            'image_id': sample['metadata']['image_id'],
            'filter_flag': True,
        }

    else:
        return key, None


def build_coco_caption_eval_datapipes_for_llm(data_dir,
                                              tokenizer=None,
                                              caption_prompt=None,
                                              recursive=True,
                                              batch_size=None,
                                              num_context=2):
    decode_partial = functools.partial(
        decode_coco_cap_eval_data,
        tokenizer=tokenizer,
        caption_prompt=caption_prompt,
        num_context=num_context,
    )
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.filter(filter_data_for_llm)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def decode_text_conversation_data_for_llm(
        item,
        tokenizer=None,
        max_length=512,
        system_message='',
        roles=('USER', 'ASSISTANT'),
        sep='\n',
):
    key, value = item
    input_ids = []
    labels = []
    input_text = ''

    if system_message != '':
        if not system_message.endswith('\n'):
            system_message += '\n'
        input_text += system_message
        item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    if value.get('data', None) is None:
        return {'filter_flag': False}

    for idx, content in enumerate(value['data']):
        # USER
        if idx % 2 == 0:
            if idx == 0:
                text = roles[idx % 2] + ': ' + content + sep + roles[(idx + 1) % 2] + ': '
            else:
                text = sep + roles[idx % 2] + ': ' + content + sep + roles[(idx + 1) % 2] + ': '
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)
        # ASSISTANT
        else:
            text = content
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = item_ids
        input_text += text
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'text': input_text,
        'filter_flag': True,
    }


def build_text_conversation_datapipes_for_llm(data_dir,
                                              tokenizer=None,
                                              max_length=512,
                                              system_message='',
                                              roles=('USER', 'ASSISTANT'),
                                              sep='\n',
                                              recursive=True,
                                              batch_size=None,
                                              cycle_count=None):
    decode_partial = functools.partial(
        decode_text_conversation_data_for_llm,
        tokenizer=tokenizer,
        max_length=max_length,
        system_message=system_message,
        roles=roles,
        sep=sep,
    )

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_data_for_llm)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def decode_visual_question_data_for_llm(item,
                                        tokenizer=None,
                                        max_length=512,
                                        system_message='',
                                        roles=('USER', 'ASSISTANT'),
                                        sep='\n'):
    key, value = item

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        if 'image_ids' not in sample:
            print('A sample has not image_ids, skip...')
            return key, {}

        input_ids = []
        labels = []
        input_text = ''

        if system_message != '':
            if not system_message.endswith('\n'):
                system_message += '\n'
            input_text += system_message
            item_ids = tokenizer.encode(system_message, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)
            input_ids.extend(item_ids)
            labels.extend(item_labels)

        image_ids_list = sample['image_ids']
        image_tokens = ''
        for image_ids in image_ids_list:
            image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN

        image_first_flag = np.random.uniform(0, 1) < 0.5
        if 'data' in sample:
            for idx, content in enumerate(sample['data']):
                # USER
                if idx % 2 == 0:
                    if idx == 0:
                        if image_first_flag:
                            text = roles[idx % 2] + ': ' + image_tokens + content + sep + roles[(idx + 1) % 2] + ': '
                        else:
                            text = roles[idx % 2] + ': ' + content + image_tokens + sep + roles[(idx + 1) % 2] + ': '
                    else:
                        text = sep + roles[idx % 2] + ': ' + content + sep + roles[(idx + 1) % 2] + ': '
                    item_ids = tokenizer.encode(text, add_special_tokens=False)
                    item_labels = [-100] * len(item_ids)
                # ASSISTANT
                else:
                    text = content
                    item_ids = tokenizer.encode(text, add_special_tokens=False)
                    item_labels = item_ids
                input_text += text
                input_ids.extend(item_ids)
                labels.extend(item_labels)
        elif 'instruction' in sample:
            instruction = sample['instruction']
            inputs = sample['inputs']
            outputs = sample['outputs']
            if inputs.strip() == '':
                inputs = instruction

            if image_first_flag:
                inputs = roles[0] + ': ' + image_tokens + inputs + sep + roles[1] + ': '
            else:
                inputs = roles[0] + ': ' + inputs + image_tokens + sep + roles[1] + ': '

            item_ids = tokenizer.encode(inputs, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)
            input_text += inputs
            input_ids.extend(item_ids)
            labels.extend(item_labels)

            item_ids = tokenizer.encode(outputs, add_special_tokens=False)
            item_labels = item_ids
            input_text += outputs
            input_ids.extend(item_ids)
            labels.extend(item_labels)
        else:
            print('Wrong data format!, skip')
            return key, {}

        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        else:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': input_text,
            'filter_flag': True,
        }

    else:
        return key, None


def build_visual_question_datapipes_for_llm(data_dir,
                                            tokenizer=None,
                                            max_length=512,
                                            system_message='',
                                            roles=('USER', 'ASSISTANT'),
                                            sep='\n',
                                            recursive=True,
                                            batch_size=None,
                                            cycle_count=None):

    decode_partial = functools.partial(
        decode_visual_question_data_for_llm,
        tokenizer=tokenizer,
        max_length=max_length,
        system_message=system_message,
        roles=roles,
        sep=sep,
    )

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
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


def decode_video_pair_for_llm(item, tokenizer=None, max_length=128, image_id_length=32, num_frames=4, flat=True):
    key, value = item

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        if 'image_ids' not in sample or 'answer' not in sample:
            return key, {}

        image_ids = sample['image_ids']
        caption = sample['answer']
        input_ids = []
        labels = []
        image_tokens = ''
        input_text = ''
        if flat:
            for i in range(num_frames):
                frame_ids = image_ids[i * image_id_length:(i + 1) * image_id_length]
                image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in frame_ids]) + EOI_TOKEN
        else:
            for frame_ids in image_ids:
                image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in frame_ids]) + EOI_TOKEN

        item_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_text += image_tokens
        input_ids.extend(item_ids)
        labels.extend(item_labels)

        item_ids = tokenizer.encode(caption, add_special_tokens=False)
        item_labels = item_ids
        input_text += caption
        input_ids.extend(item_ids)
        labels.extend(item_labels)

        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        else:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': input_text,
            'filter_flag': True,
        }

    else:
        return key, None


def build_video_caption_datapipes_for_llm(data_dir,
                                          tokenizer=None,
                                          max_length=128,
                                          image_id_length=32,
                                          num_frames=4,
                                          flat=True,
                                          recursive=True,
                                          batch_size=None,
                                          cycle_count=None):

    decode_partial = functools.partial(
        decode_video_pair_for_llm,
        tokenizer=tokenizer,
        max_length=max_length,
        image_id_length=image_id_length,
        num_frames=num_frames,
        flat=flat,
    )

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
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


def decode_text_pretrain_data_for_llm(item, tokenizer=None, max_length=512):
    key, value = item
    input_ids = []
    labels = []
    input_text = ''

    text = value.get('text', None)
    if text is None:
        return {'filter_flag': False}

    filter_flag = True
    if text.strip(' ;,[]{}\'\".?:') == '':
        filter_flag = False

    if tokenizer is None:
        return {'text': text, 'filter_flag': filter_flag}

    tokenized = tokenizer(tokenizer.bos_token + text + tokenizer.eos_token,
                          max_length=max_length,
                          add_special_tokens=False,
                          truncation=True,
                          padding='max_length',
                          return_tensors='pt')

    input_ids = tokenized['input_ids'][0]
    attention_mask = tokenized['attention_mask'][0]
    labels = torch.clone(input_ids)
    labels[labels == tokenizer.pad_token_id] = -100
    # print('labels before: ',labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'text': text,
        'filter_flag': filter_flag,
    }


def build_text_pretrain_datapipes_for_llm(data_dir,
                                          tokenizer=None,
                                          max_length=512,
                                          recursive=True,
                                          batch_size=None,
                                          cycle_count=None):
    decode_partial = functools.partial(decode_text_pretrain_data_for_llm, tokenizer=tokenizer, max_length=max_length)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_data_for_llm)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe