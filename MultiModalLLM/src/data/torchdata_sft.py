import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import pickle
import random
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from braceexpand import braceexpand
import hydra

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

gen_prompt = [
"Please show me a picture of",
"Please design an image of",
"Please produce a photo of",
"Please generate an image of",
"Please draw a painting of",
"I'd like to see a drawing of",
"I'd love to see an illustration of",
"I'd like to view an image of",
"I want to see a picture of",
"I would like to see a photo of",
"Show me a photo of",
"Generate a picture of",
"Show me a photograph of",
"Generate an image of",
"Can you make an image of",
"Can you draw a painting of",
"Can you produce a picture of",
"Can you generate a photo of",
"Can you depict a picture of",
"Can you show me an illustration of"]

gen_prompt_response = [
"Here is a picture.",
"I have designed an image.",
"Here is a photo.",
"I have generated an image.",
"Here's a painting.",
"Here's a drawing.",
"Enjoy this illustration.",
"Take a look at this image.",
"Here is a picture.",
"I have created a photo.",
"Enjoy this photo.",
"I have generated a picture.",
"Here is a photograph.",
"Here's an image.",
"Certainly, here's an image.",
"Absolutely, here is a painting.",
"Sure, here is a picture.",
"Of course, here is a photo.",
"Certainly, please enjoy this picture.",
"Sure, please enjoy this illustration."]

edit_prompt = [
"Please show me a picture.",
"Please create an image.",
"Please produce a picture.",
"Please generate an image.",
"Generate a picture, please.",
"Create an image, please.",
"Can you generate an image?",
"Can you produce a picture?"]

edit_prompt_response = [
"Here is a picture.",
"I have created an image.",
"Enjoy this picture.",
"I have generated an image.",
"Here's a picture.",
"Here's an image.",
"Certainly, here's an image.",
"Sure, here is a picture.",]

story_prompt = [
"Can you use images and texts to tell me what will happen next?",
"Would you utilize pictures and texts to show me the following story?",
"Can you show me with images and texts what will occur then?",
"Can you tell me the following event using images and written descriptions?",
"Can you use images and texts to express to me the forthcoming happenings?",
"Please use images and text to inform me what will happen next.",
"Through pictures and text, let me know what will happen later.",
"Show me with images and texts what will occur then.",
"Please utilize pictures and descriptions to tell me what will happen subsequently.",
"Please generate the following story using both images and texts."
]

story_prompt_response = [
"Sure, let me tell you what will happen next using images and texts.",
"I would be happy to utilize pictures and texts to show you the following story.",
"Certainly, I can show you with images and texts what will occur then.",
"Of course! Let me tell you the following event using images and written descriptions.",
"Sure, here are images and texts to let you know the forthcoming happenings.",
"Here are images and text to inform you what will happen next.",
"Please see the pictutres and text to know what will happen later.",
"Here are images and texts for you to know what will occur then.",
"Let me show you what will happen subsequently with pictures and descriptions.",
"I have generated the following story using both images and texts."
]

s_token = "USER:"
e_token = "ASSISTANT:"


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
                text = s_token + " " + content + sep + e_token
            else:
                text = sep + s_token + " " + content + sep + e_token
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)
        # ASSISTANT
        else:
            text = content
            #print(text)
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

        # system_message = 'You will be presented with an image.'
        # if system_message != '':
        #     if not system_message.endswith('\n'):
        #         system_message += '\n'
        #     input_text += system_message
        #     item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        #     item_labels = [-100] * len(item_ids)
        #     input_ids.extend(item_ids)
        #     labels.extend(item_labels)

        image_ids_list = sample['image_ids']
        image_tokens = ''
        for image_ids in image_ids_list:
            image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN

        if 'data' in sample:

            for idx, content in enumerate(sample['data']):
                content = content.strip()
                # USER
                if idx % 2 == 0:
                    if idx == 0:
                        text = s_token + " " + image_tokens + content + sep + e_token
                    else:
                        text = sep + s_token + " " + content + sep + e_token
                    item_ids = tokenizer.encode(text, add_special_tokens=False)
                    item_labels = [-100] * len(item_ids)
                # ASSISTANT
                else:
                    text = content
                    if not text.endswith('.'):
                        text = text + '.'
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

            inputs = s_token + " " + image_tokens + inputs + sep + e_token

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
    
def decode_edit_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.5):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        if 'source_image_ids' in sample.keys():
            source_image_ids = sample['source_image_ids']
            target_image_ids = sample['target_image_ids']
            
            source_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in source_image_ids]) + EOI_TOKEN
            target_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in target_image_ids]) + EOI_TOKEN

            if 'instruction_gpt' in sample.keys():
                instruction_gpt = sample['instruction_gpt']
                response = sample['response']
                system_message = ''                
                question = system_message + s_token + " " + source_image_tokens + instruction_gpt + sep + e_token
                answer = response + target_image_tokens
            else:
                instruction = sample['instruction']
                # if not instruction.endswith('.'):
                #     instruction = instruction + '.'

                indx = random.randint(0, 5)
                prompt = edit_prompt[indx]
                response = edit_prompt_response[indx]

                system_message = ''      
                question = system_message + s_token + " " + source_image_tokens + instruction + sep + e_token
                answer = response + target_image_tokens

            question_ids = tokenizer.encode(question, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                
            labels = [-100] * len(question_ids) + answer_ids
            input_ids = question_ids + answer_ids
            
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = [-100] + labels + [tokenizer.eos_token_id]

            #print(len(input_ids))
            
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
            # 'text': caption,
        }
    else:
        return key, None


def decode_edit_multi_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.5):
    key, value = item
    sep = '\n'

    input_ids = []
    labels = []

    if key.endswith(".pkl"):
        sample = pickle.load(value)

        if 'image_ids' in sample.keys():
            
            image_ids = sample['image_ids']
            instruction0 = sample['instruction0']
            instruction1 = sample['instruction1']
            instruction2 = sample['instruction2']

            image_tokens0 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*0:32*1]]) + EOI_TOKEN
            image_tokens1 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*1:32*2]]) + EOI_TOKEN
            image_tokens2 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*2:32*3]]) + EOI_TOKEN
            image_tokens3 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*3:32*4]]) + EOI_TOKEN
        

            indx = random.randint(0, 5)
            response = edit_prompt_response[indx]

            system_message = ''      
            q = system_message + s_token + " " + image_tokens0 + instruction0 + sep + e_token
            a = response + image_tokens1

            q_ids = tokenizer.encode(q, add_special_tokens=False)
            a_ids = tokenizer.encode(a, add_special_tokens=False)
            
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids

            input_ids.extend(input_ids_item)
            labels.extend(labels_item)   
            
            q = sep + s_token + " " + instruction1 + sep + e_token
            a = response + image_tokens2

            q_ids = tokenizer.encode(q, add_special_tokens=False)
            a_ids = tokenizer.encode(a, add_special_tokens=False)
            
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids

            input_ids.extend(input_ids_item)
            labels.extend(labels_item)   

            q = sep + s_token + " " + instruction2 + sep + e_token
            a = response + image_tokens3

            q_ids = tokenizer.encode(q, add_special_tokens=False)
            a_ids = tokenizer.encode(a, add_special_tokens=False)
            
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids

            input_ids.extend(input_ids_item)
            labels.extend(labels_item)   

            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = [-100] + labels + [tokenizer.eos_token_id]
            
            #print(len(input_ids))
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
                # 'text': caption,
            }
    else:
        return key, None
    
    

def decode_conversation_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.5):
    key, value = item
    sep = '\n'
    
    if key.endswith(".pkl"):
        sample = pickle.load(value)

        question = sample['question']
        answer = sample['answer']
        image_ids = sample['image_ids']
        
        input_ids = []
        labels = []

        # system_message = 'You will be presented with an image.'
        # if system_message != '':
        #     if not system_message.endswith('\n'):
        #         system_message += '\n'
        #     item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        #     item_labels = [-100] * len(item_ids)
        #     input_ids.extend(item_ids)
        #     labels.extend(item_labels)
                    
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN

        if True:
            if 'Image Descriptions' not in question:
                question_list = question.split('Question: ')[1:]
                answer_list = answer.split('Answer: ')[1:]
                num_qa = np.minimum(len(question_list), len(answer_list))
  
                for i in range(num_qa):
                    q = question_list[i].strip().replace('Question: ', '')
                    a = answer_list[i].strip().replace('Answer: ', '')
                    if not a.endswith('.'):
                        a = a + '.'
                    
                    if i == 0:
                        q = s_token + " " + image_tokens + q + sep + e_token
                    else:
                        q = sep + s_token + " " + q + sep + e_token
                        
                    q_ids = tokenizer.encode(q, add_special_tokens=False)
                    a_ids = tokenizer.encode(a, add_special_tokens=False)
                    
                    labels_item = [-100] * len(q_ids) + a_ids
                    input_ids_item = q_ids + a_ids

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
        
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    else:
        return key, None

def decode_video_conversation_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.5):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)

        question = sample['question']
        answer = sample['answer']
        image_ids = sample['image_ids']
        
        input_ids = []
        labels = []

        # system_message = 'You will be presented with a video consisting of multiple images.'
        # if system_message != '':
        #     if not system_message.endswith('\n'):
        #         system_message += '\n'
        #     item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        #     item_labels = [-100] * len(item_ids)
        #     input_ids.extend(item_ids)
        #     labels.extend(item_labels)
            
        num_frames = int(len(image_ids) / 32)
        image_id_length = 32
        image_tokens = ''
        for i in range(num_frames):
            frame_ids = image_ids[i * image_id_length:(i + 1) * image_id_length]
            image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in frame_ids]) + EOI_TOKEN
    
        if 'Question:' in question and 'Answer:' in answer:
            question_list = question.split('Question: ')[1:]
            answer_list = answer.split('Answer: ')[1:]
            assert len(question_list) == len(answer_list)
            num_qa = len(question_list)
            for i in range(num_qa):

                q = question_list[i].strip().replace('Question: ', '')
                a = answer_list[i].strip().replace('Answer: ', '')
                
                if i == 0:
                    q = s_token + " " + image_tokens + q + sep + e_token
                else:
                    q = sep + s_token + " " + q + sep + e_token
                
                q_ids = tokenizer.encode(q, add_special_tokens=False)
                a_ids = tokenizer.encode(a, add_special_tokens=False)
                
                labels_item = [-100] * len(q_ids) + a_ids
                input_ids_item = q_ids + a_ids

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
        
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    else:
        return key, None


def decode_difference_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.0):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        image_ids = sample['image_ids']
        question = sample['question'].strip()
        question = question.replace('image a', 'the first image')
        question = question.replace('image A', 'the first image')
        question = question.replace('image b', 'the second image')
        question = question.replace('image B', 'the second image')
        answer = sample['answer'].strip()
        answer = answer.replace('image a', 'the first image')
        answer = answer.replace('image A', 'the first image')
        answer = answer.replace('image b', 'the second image')
        answer = answer.replace('image B', 'the second image')
        if not answer.endswith('.'):
            answer = answer + '.'
        image_tokens0 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[0:32]]) + EOI_TOKEN
        image_tokens1 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32:64]]) + EOI_TOKEN
        
        if reverse_ratio == 0.0 or np.random.uniform(0, 1) < reverse_ratio:
            # system_message = 'You will be presented with an image.'
            # if system_message != '':
            #     if not system_message.endswith('\n'):
            #         system_message += '\n'
            system_message = ''
 
            if np.random.uniform(0, 1) < 0.5:
                question = system_message + s_token + " " + image_tokens0 + 'This is the first image.' + image_tokens1 + 'This is the second image. ' + question + sep + e_token
            else:
                question = system_message + s_token + " " + 'This is the first image.' + image_tokens0 + 'This is the second image.' + image_tokens1 + question + sep + e_token
            question_ids = tokenizer.encode(question, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                
            labels = [-100] * len(question_ids) + answer_ids
            input_ids = question_ids + answer_ids
            
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = [-100] + labels + [tokenizer.eos_token_id]

            #print(len(input_ids))
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
            # 'text': caption,
        }
    else:
        return key, None


def decode_story_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.0):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        image_ids = sample['image_ids']
        caption0 = sample['caption0'].strip()
        caption1 = sample['caption1'].strip()
        caption2 = sample['caption2'].strip()
        caption3 = sample['caption3'].strip()
        caption4 = sample['caption4'].strip()

        image_tokens0 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*0:32*1]]) + EOI_TOKEN
        image_tokens1 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*1:32*2]]) + EOI_TOKEN
        image_tokens2 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*2:32*3]]) + EOI_TOKEN
        image_tokens3 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*3:32*4]]) + EOI_TOKEN
        image_tokens4 = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids[32*4:32*5]]) + EOI_TOKEN

        input_ids = []
        labels = []
        if reverse_ratio == 0.0:

            system_message = ''
            response = 'Here is an image.'
            
            question = system_message + s_token + " " + caption0 + image_tokens0 + caption1 + ' Please generate an image.' + sep + e_token
            answer = response + image_tokens1
            
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids
            input_ids.extend(input_ids_item)
            labels.extend(labels_item)
                
            question = sep + s_token + " " + caption2 + ' Please generate an image.' + sep + e_token
            answer = response + image_tokens2
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids
            input_ids.extend(input_ids_item)
            labels.extend(labels_item)
            
            question = sep + s_token + " " + caption3 + ' Please generate an image.' + sep + e_token
            answer = response + image_tokens3
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids
            input_ids.extend(input_ids_item)
            labels.extend(labels_item)
            
            question = sep + s_token + " " + caption4 + ' Please generate an image.' + sep + e_token
            answer = response + image_tokens4
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids
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
        else:
            system_message = ''
            
            indx = random.randint(0, 9)
            prompt = story_prompt[indx]
            response = story_prompt_response[indx]
                
            question = system_message + s_token + " " + caption0 + image_tokens0 + prompt + sep + e_token
            answer = response + sep + caption1 + image_tokens1 + sep + caption2 + image_tokens2 + sep + caption3 + image_tokens3 + sep + caption4 + image_tokens4
            
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids + a_ids
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
            
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            # 'text': caption,
        }
    else:
        return key, None
    

def decode_question_answer_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.0):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        image_ids = sample['image_ids']
        if 'question' in sample.keys():
            question = sample['question']
            answer = sample['answer']
            question = question.strip()
        else:
            question = None 
            answer = sample['text']
        answer = answer.strip()

        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN
            
        if reverse_ratio == 0.0 or np.random.uniform(0, 1) < reverse_ratio:
            if not answer.endswith('.'):
                answer = answer + '.'

            system_message = ''
                
            question = system_message + s_token + " " + image_tokens + question + sep + e_token
            
            question_ids = tokenizer.encode(question, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                
            labels = [-100] * len(question_ids) + answer_ids
            input_ids = question_ids + answer_ids
            
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
        
        else:
            if answer.endswith('.'):
                answer = answer.rstrip(".")

            system_message = ''
                
            indx = random.randint(0, 19)
            prompt = gen_prompt[indx]
            response = gen_prompt_response[indx]
            if indx >= 14:
                punct = '?'
            else:
                punct = '.'
            question = system_message + s_token + " " + prompt + ' ' + answer + punct + sep + e_token
            answer = response + image_tokens

            question_ids = tokenizer.encode(question, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                
            labels = [-100] * len(question_ids) + answer_ids
            input_ids = question_ids + answer_ids
            
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = [-100] + labels + [tokenizer.eos_token_id]
            #print(len(input_ids))
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
            # 'text': caption,
        }
    else:
        return key, None




def decode_interleaved_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.0):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        instruction = sample['instruction']
        answer = sample['response']

        system_message = ''
            
        question = system_message + s_token + " " + instruction + sep + e_token
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        #print(len(input_ids))
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
            # 'text': caption,
        }
    else:
        return key, None
    
def decode_video_pair_for_llm(item, tokenizer=None, max_length=128, caption_prompt=None, reverse_ratio=0.5):
    key, value = item
    sep = '\n'

    if key.endswith(".pkl"):
        sample = pickle.load(value)
        image_ids = sample['image_ids']
        question = sample['question']
        answer = sample['answer']

        question = question.strip()
        answer = answer.strip()
        if '\nQA_GT_caption_based_noisy' in answer:
             answer = answer.replace('\nQA_GT_caption_based_noisy', '')

        if not answer.endswith('.'):
            answer = answer + '.'
        num_frames = int(len(image_ids) / 32)
        image_id_length = 32
        image_tokens = ''
        for i in range(num_frames):
            frame_ids = image_ids[i * image_id_length:(i + 1) * image_id_length]
            image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in frame_ids]) + EOI_TOKEN
        system_message = ''
        
        question = system_message + s_token + " " + image_tokens + question + sep + e_token
        
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        
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
    return unwarpped


def filter_data_for_llm(item):
    if 'input_ids' in item:
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

def build_edit_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_edit_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_edit_multi_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_edit_multi_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_conversation_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_conversation_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_video_conversation_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_video_conversation_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe

def build_qa_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.0,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_question_answer_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe

def build_interleaved_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.0,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_interleaved_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe

def build_video_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_video_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_video_gen_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_video_gen_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe

def build_difference_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_difference_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_story_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(decode_story_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe

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

