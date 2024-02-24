import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import os
from random import choice
from braceexpand import braceexpand
import cv2
import random
import numpy as np
import torch

def decode(item, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
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
        return key, {'metadata': value.read().decode('utf-8')}
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
    if ('pixel_values' in item and 'text' in item) or ('question' in item and 'answer' in item and 'pixel_values' in item)\
        or ('instruction' in item and 'source_pixel_values' in item and 'target_pixel_values' in item):
        # if 'on top of the back of' in item['answer']:
        #     print('filtered')
        #     return False
        # else:
        return True
    else:
        print('filtered')
        return False


def build_caption_datapipes_with_pixels(data_dir,
                                        tokenizer=None,
                                        image_processor=None,
                                        image_transform=None,
                                        max_length=128,
                                        batch_size=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)

    if isinstance(data_dir, str):
        data_dir = braceexpand(data_dir)
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data)
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


@dp.functional_datapipe("parse_json_list")
class JsonListParser(dp.iter.IterDataPipe):

    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for _, json_data in self.source_datapipe:
            for item in json_data:
                if ('caption' in item and 'image' in item) or ('question' in item and 'answer' in item  and 'image' in item) or \
                    ('instruction' in item and 'source_image' in item and 'target_image' in item):
                    yield item
                if 'video' in item or 'video_id' in item:
                    # num = len(item['QA'])
                    # if num == 1:
                    yield item



def decode_coco_image_text_pair(item, image_root, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    caption = item['caption']
    image_name = item['image']
    image_path = os.path.join(image_root, image_name)
    image = Image.open(image_path).convert('RGB')

    metadata = {'image_id': item['image_id']}
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

def decode_question_answer_pair(item, image_root, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    ## coco
    # image_dir = 'coco/'
    # image_path = item['image']
    # image_path = image_dir + image_path
    # question = 'Please provide an accurate and concise description of the given image.'
    # answer = item['caption']
    # if isinstance(answer, list):
    #     answer = choice(answer)

    ## VQAv2
    # image_dir = 'coco/'
    # image_path = item['image']
    # image_path = image_dir + image_path
    # question = item['question'] + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    # if isinstance(answer, list):
    #     answer = choice(answer)
        
    ## OK-VQA    
    # image_path = item['image']
    # question = item['question'] + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    
    # VizWiz
    # question = item['question'] + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    # image_path = item['image']
    
    # VisDial
    # system = 'Based on the image caption and dialog history, answer the following question.\n'
    # question = system + item['question']
    # answer = item['answer']
    # image_path = item['image']
    
    # llava conversation, SVIT
    # question = item['question']
    # answer = item['answer']
    # image_path = item['image']

    # JourneyDB
    question = 'Please provide an accurate and concise description of the given image.'
    answer = item['caption']
    image_path = item['image']
    
    #print(item.keys())
    # question = item['question']
    # answer = item['answer']
    # image_path = item['image']
    
    # question = item['question'] + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    # image_path = item['image']
    
    # system = 'Based on the image caption and dialog history, please provide an accurate answer consisting of only one word or phrase. '
    # question = system + item['question']
    # answer = item['answer']
    # image_path = item['image']
    
    # question = question + ' Please answer this question briefly.'
    # if isinstance(answer, list):
    #     answer = choice(answer)
    # #print(question, answer)
    
    # question = "Generate a caption for this image briefly."
    # answer =  item['caption']
    # image_path = item['image']
    
    # if not os.path.exists(image_path):
    #     sample = {'error': True}
    #     return sample
    try:  
        image = Image.open(image_path).convert('RGB')
    except:
        sample = {'error': True}
        return sample
    
    sample = {'question': question, 'answer': answer}

    #print('transform', image_processor, image_transform)
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


def decode_edit_pair(item, image_root, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    
    ## instruct-pix2pix, MagicBrush
    instruction = item['instruction']
    source_image_path = item['source_image']
    target_image_path = item['target_image']
    source_image = Image.open(source_image_path).convert('RGB')
    target_image = Image.open(target_image_path).convert('RGB')
    
    sample = {'instruction': instruction}
    
    #caption = item['caption']
    #caption = item['instruction']
    #instruction = 'Edit this image with the instruction: ' + item['instruction']
    
    #print('transform', image_processor, image_transform)
    if image_processor is not None:
        try:
            image_tensor = image_processor(source_image, return_tensors='pt')['pixel_values'][0]
            sample['source_pixel_values'] = image_tensor
            image_tensor = image_processor(target_image, return_tensors='pt')['pixel_values'][0]
            sample['target_pixel_values'] = image_tensor
        except Exception as e:
            print('Error while process image: ', e)
    elif image_transform is not None:
        try:
            image_tensor = image_transform(source_image)
            sample['source_pixel_values'] = image_tensor
            image_tensor = image_transform(target_image)
            sample['target_pixel_values'] = image_tensor
        except Exception as e:
            print('Error while transform image: ', e)
    else:
        sample['source_pixel_values'] = source_image
        sample['target_pixel_values'] = target_image

    return sample

def sample_frames(num_frames, vlen, sample='uniform', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample, fix_start=None):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    #frames = torch.stack(frames).float()
    cap.release()
    return frames, success_idxs


def decode_video_pair(item, image_root, tokenizer=None, image_processor=None, image_transform=None, max_length=128):
    # Video-ChatGPT
    # question = item['question']
    # answer = item['answer']
    # video_path = item['video']
    # num_frames = 8
    
    # NextQA
    # question = item['question']
    # if not question.endswith('?'):
    #     question = question + '?'
    # question = question + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    # video_path = item['video']
    # num_frames = 8
    
    # MSR-VTT caption
    # question = "Please provide an accurate and concise description of the given video."
    # answer = item['caption']
    # if isinstance(answer, list):
    #     answer = choice(answer) 
    # video_path = item['video']       
    # num_frames = 4
        
    # # MSR-VTT qa 
    # questions = item['question']
    # answers = item['answer']
    # if isinstance(answers, list):
    #     num_qa = len(questions)
    #     indx = random.randint(0, num_qa - 1)
    #     question = questions[indx] +  ' Please provide an accurate answer consisting of only one word or phrase.'
    #     answer = answers[indx]
    # video_path = item['video']       
    # num_frames = 4
        
    # # MSVD caption
    # question = "Please provide an accurate and concise description of the given video."
    # answer = item['caption']
    # video_path = item['video']       
    # num_frames = 4
        
    # # MSVD qa 
    # question = item['question'] + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    # video_path = item['video']       
    # num_frames = 4
    
    # # Webvid
    question = "Please provide an accurate and concise description of the given video."
    answer = item['caption']
    video_path = item['video']       
    num_frames = 4
    

    #video_path = item['video']    
    # question = "Please provide an accurate and concise description of the given video."
    
    # QA = item['QA']
    # num = len(QA)
    # question_concat = ''
    # answer_concat = ''
    # for i in range(num):
    #     question = QA[i]['q']
    #     answer = QA[i]['a']
    #     question_concat = question_concat + 'Question: ' + question + ' '
    #     answer_concat = answer_concat + 'Answer: ' + answer + ' '
        
    # question = question_concat.strip()
    # answer = answer_concat.strip()
    
    # assert len(QA) == 1
    # question = QA[0]['q']
    # answer = QA[0]['a']
    #print(question, answer)

    # question = item['question']
    # if not question.endswith('?'):
    #     question = question + '?'
    # question = question + ' Please provide an accurate answer consisting of only one word or phrase.'
    # answer = item['answer']
    # video_path = item['video']
    
    # question = item['q']
    # answer = item['a']
    # path = item['video_id']
    # question = "Generate a caption for this video briefly."
    # answer = item['caption']
    # if isinstance(answer, list):
    #     answer = choice(answer) 
    #     print(answer)
        
    # questions = item['question']
    # answers = item['answer']
    # if isinstance(answers, list):
    #     num_qa = len(questions)
    #     indx = random.randint(0, num_qa - 1)
    #     question = questions[indx] + ' Please answer this question briefly.'
    #     answer = answers[indx]
    #     print(question, answer)
        
    # video_path = item['video']       
    
    #video_path = item['video']    
    # question = "Please provide an accurate and concise description of the given video."
    # answer = item['caption']
    #question = item['question']
    #answer = item['answer']
    #question = question + ' Please provide an accurate answer consisting of only one word or phrase.'
    
    try:
        assert os.path.exists(video_path)
        imgs, idxs = read_frames_cv2(video_path, num_frames=num_frames, sample='uniform')
        assert imgs.shape[0] == num_frames
        assert imgs.shape[1] == 3
        sample = {'question': question, 'answer': answer}

        #print('transform', image_processor, image_transform)
        if image_processor is not None:
            try:
                image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]
                sample['pixel_values'] = image_tensor
            except Exception as e:
                print('Error while process image: ', e)
        elif image_transform is not None:
            try:
                image_tensor = image_transform(imgs)
                sample['pixel_values'] = image_tensor
                #print(imgs.shape, image_tensor.shape)
            except Exception as e:
                print('Error while transform image: ', e)
        else:
            sample['pixel_values'] = image
    except:
        sample = {'error': True}

    return sample


from torchdata.datapipes.iter import ShardingFilter


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
    datapipe = datapipe.parse_json_list()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)

    datapipe = datapipe.filter(filter_data)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
    return datapipe

def build_question_answer_datapipes(data_dir,
                                 mask=None,
                                 recursive=False,
                                 tokenizer=None,
                                 image_processor=None,
                                 image_transform=None,
                                 max_length=128,
                                 batch_size=None):

    decode_partial = functools.partial(decode_question_answer_pair,
                                       image_root=data_dir,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)

    mask = '*.json' if mask is None else mask
    datapipe = dp.iter.FileLister(root=data_dir, masks=mask, recursive=recursive)
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_json_files()
    datapipe = datapipe.parse_json_list()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)

    datapipe = datapipe.filter(filter_data)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
    return datapipe


def build_edit_datapipes(data_dir,
                                 mask=None,
                                 recursive=False,
                                 tokenizer=None,
                                 image_processor=None,
                                 image_transform=None,
                                 max_length=128,
                                 batch_size=None):

    decode_partial = functools.partial(decode_edit_pair,
                                       image_root=data_dir,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)

    mask = '*.json' if mask is None else mask
    datapipe = dp.iter.FileLister(root=data_dir, masks=mask, recursive=recursive)
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_json_files()
    datapipe = datapipe.parse_json_list()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)

    datapipe = datapipe.filter(filter_data)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
    return datapipe

def build_video_datapipes(data_dir,
                                 mask=None,
                                 recursive=False,
                                 tokenizer=None,
                                 image_processor=None,
                                 image_transform=None,
                                 max_length=128,
                                 batch_size=None):

    decode_partial = functools.partial(decode_video_pair,
                                       image_root=data_dir,
                                       tokenizer=tokenizer,
                                       image_processor=image_processor,
                                       image_transform=image_transform,
                                       max_length=max_length)

    mask = '*.json' if mask is None else mask
    datapipe = dp.iter.FileLister(root=data_dir, masks=mask, recursive=recursive)
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_json_files()
    datapipe = datapipe.parse_json_list()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)

    datapipe = datapipe.filter(filter_data)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
    return datapipe


