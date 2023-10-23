import hydra

import pyrootutils
import os
import torch

import datetime
from omegaconf import OmegaConf
# from flask import Flask, request
import json
from typing import Optional
import transformers
from dataclasses import dataclass, field
import io
import base64
from PIL import Image
import gradio as gr
import random
import time
import hashlib
import requests

from utils import build_logger
from conversation import conv_seed_vicuna, conv_seed_llama2
# from conversation import conv_seed_llama

IMG_FLAG = '<image>'

# request_address = 'http://11.29.21.161:80/generate'
# request_address = 'http://0.0.0.0:7890/generate'
LOGDIR = 'log'

logger = build_logger("gradio_seed_llama", LOGDIR)
headers = {"User-Agent": "SEED LLaMA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

@dataclass
class Arguments:
    server_port: Optional[int] = field(default=7860, metadata={"help": "network port"})
    server_name: Optional[str] = field(default='0.0.0.0', metadata={"help": "network address"})
    request_address: Optional[str] = field(default='http://127.0.0.1:7890/generate', metadata={"help": "request address"})
    model_type: Optional[str] = field(default='seed-llama-14b', metadata={"help": "choice: [seed-llama-8b, seed-llama-14b]"})

parser = transformers.HfArgumentParser(Arguments)
args, = parser.parse_args_into_dataclasses()

if args.model_type == 'seed-llama-8b':
    conv_seed_llama = conv_seed_vicuna
elif args.model_type == 'seed-llama-14b':
    conv_seed_llama = conv_seed_llama2
else:
    raise ValueError


def decode_image(encoded_image: str) -> Image:
    decoded_bytes = base64.b64decode(encoded_image.encode('utf-8'))
    # with io.BytesIO(decoded_bytes) as buffer:
    #     image = Image.open(buffer)
    #     return image
    buffer = io.BytesIO(decoded_bytes)
    image = Image.open(buffer)
    return image


def encode_image(image: Image.Image, format: str = 'PNG') -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_conv_image_dir():
    name = os.path.join(LOGDIR, 'images')
    os.makedirs(name, exist_ok=True)
    return name


def get_image_name(image, image_dir=None):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    md5 = hashlib.md5(image_bytes).hexdigest()

    if image_dir is not None:
        image_name = os.path.join(image_dir, md5 + '.png')
    else:
        image_name = md5 + '.png'

    return image_name


def resize_image(image, max_size=512):
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    if width > height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    return resized_image


def center_crop_image(image, max_aspect_ratio=1.5):
    width, height = image.size
    aspect_ratio = max(width, height) / min(width, height)

    if aspect_ratio >= max_aspect_ratio:
        if width > height:
            new_width = int(height * max_aspect_ratio)
            left = (width - new_width) // 2
            right = (width + new_width) // 2
            top = 0
            bottom = height
        else:
            new_height = int(width * max_aspect_ratio)
            left = 0
            right = width
            top = (height - new_height) // 2
            bottom = (height + new_height) // 2

        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image
    else:
        return image

def vote_last_response(state, vote_type, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", request)
    return (disable_btn, ) * 2


def downvote_last_response(state, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", request)
    return (disable_btn, ) * 2


def regenerate(dialog_state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    if dialog_state.messages[-1]['role'] == dialog_state.roles[1]:
        dialog_state.messages.pop()
    return (
        dialog_state,
        dialog_state.to_gradio_chatbot(),
    ) + (disable_btn, ) * 4


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    # state = None
    # return (state, [], "") + (disable_btn, ) * 5
    dialog_state = conv_seed_llama.copy()
    input_state = init_input_state()
    return (dialog_state, input_state, dialog_state.to_gradio_chatbot()) + (disable_btn, ) * 4


def init_input_state():
    return {'images': [], 'text': '', 'images_ids': []}


def add_text(dialog_state, input_state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}.")
    # if len(input_state['text']) == 0:
    if text is None or len(text) == 0:
        # dialog_state.skip_next = True
        return (dialog_state, input_state, "", dialog_state.to_gradio_chatbot()) + (no_change_btn, ) * 4
    input_state['text'] += text

    # dialog_state.skip_next = False

    if len(dialog_state.messages) > 0 and dialog_state.messages[-1]['role'] == dialog_state.roles[0]:
        dialog_state.messages[-1]['message'] = input_state
    else:
        dialog_state.messages.append({'role': dialog_state.roles[0], 'message': input_state})
    print('add_text: ', dialog_state.to_gradio_chatbot())

    return (dialog_state, input_state, "", dialog_state.to_gradio_chatbot()) + (disable_btn, ) * 4


def add_image(dialog_state, input_state, image, request: gr.Request):
    logger.info(f"add_image. ip: {request.client.host}.")
    if image is None:
        return (dialog_state, input_state, None, dialog_state.to_gradio_chatbot()) + (no_change_btn, ) * 4

    image = image.convert('RGB')
    image = resize_image(image, max_size=512)
    image = center_crop_image(image, max_aspect_ratio=1.3)
    image_dir = get_conv_image_dir()
    image_path = get_image_name(image=image, image_dir=image_dir)
    if not os.path.exists(image_path):
        image.save(image_path)

    input_state['images'].append(image_path)
    input_state['text'] += IMG_FLAG
    input_state['images_ids'].append(None)

    if len(dialog_state.messages) > 0 and dialog_state.messages[-1]['role'] == dialog_state.roles[0]:
        dialog_state.messages[-1]['message'] = input_state
    else:
        dialog_state.messages.append({'role': dialog_state.roles[0], 'message': input_state})

    print('add_image:', dialog_state)

    return (dialog_state, input_state, None, dialog_state.to_gradio_chatbot()) + (disable_btn, ) * 4


def http_bot_test(dialog_state, input_state, temperature, top_p, max_new_tokens, num_beams, max_turns, force_image_gen, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    output_state = {}
    output_state['text'] = 'This is test for frontend!'
    output_state['images'] = []
    if len(dialog_state.messages) > 0 and len(dialog_state.messages[-1]['message']['images']) != 0:
        image = random.choice(dialog_state.messages[-1]['message']['images'])
        output_state['images'].append(image)
        output_state['text'] += IMG_FLAG

    dialog_state.messages.append({'role': dialog_state.roles[1], 'message': output_state})
    input_state = init_input_state()

    print('http_bot: ', dialog_state.to_gradio_chatbot())

    return (dialog_state, input_state, dialog_state.to_gradio_chatbot()) + (enable_btn, ) * 4


def update_error_msg(chatbot, error_msg):
    if len(error_msg) > 0:
        info = '\n-------------\nSome errors occurred during response, please clear history and restart.\n' + '\n'.join(
            error_msg)
        chatbot[-1][-1] = chatbot[-1][-1] + info

    return chatbot


def http_bot(dialog_state, input_state, temperature, top_p, max_new_tokens, num_beams, max_turns, force_image_gen, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    print('input_state:', input_state)

    if len(dialog_state.messages) == 0 or dialog_state.messages[-1]['role'] != dialog_state.roles[0] or len(
            dialog_state.messages[-1]['message']['text'].strip(' ?.;!/')) == 0:
        # if len(input_state['text']) == 0:
        # dialog_state.skip_next = True
        return (dialog_state, input_state, dialog_state.to_gradio_chatbot()) + (no_change_btn, ) * 4

    if len(dialog_state.messages) > max_turns * 2:
        output_state = init_input_state()
        output_state['text'] = 'Error: History exceeds maximum rounds, please clear history and restart.'
        dialog_state.messages.append({'role': dialog_state.roles[1], 'message': output_state})
        input_state = init_input_state()
        return (dialog_state, input_state, dialog_state.to_gradio_chatbot()) + (disable_btn, ) * 3 + (enable_btn, )

    prompt = dialog_state.get_prompt()
    payload = {
        'text': prompt['text'],
        'temperature': float(temperature),
        'top_p': float(top_p),
        'max_new_tokens': int(max_new_tokens),
        'num_beams': int(num_beams),
        'images': prompt['images'],
        'force_boi': force_image_gen,
    }

    print(
        'request: ', {
            'text': prompt['text'],
            'temperature': float(temperature),
            'top_p': float(top_p),
            'max_new_tokens': int(max_new_tokens),
            'num_beams': int(num_beams)
        })
    print('request_address', args.request_address)
    response = requests.request(method="POST", url=args.request_address, headers=headers, json=payload)
    results = response.json()
    print('response: ', {'text': results['text'], 'images_ids': results['images_ids'], 'error_msg': results['error_msg']})

    output_state = init_input_state()
    image_dir = get_conv_image_dir()
    output_state['text'] = results['text']

    for image_base64 in results['images']:
        if image_base64 == '':
            image_path = ''
        else:
            image = decode_image(image_base64)
            image = image.convert('RGB')
            image_path = get_image_name(image=image, image_dir=image_dir)
            if not os.path.exists(image_path):
                image.save(image_path)
        output_state['images'].append(image_path)
        output_state['images_ids'].append(None)

    dialog_state.messages.append({'role': dialog_state.roles[1], 'message': output_state})
    dialog_state.update_image_ids(results['images_ids'])
    
    vote_last_response(dialog_state, 'common', request)
    input_state = init_input_state()
    chatbot = update_error_msg(dialog_state.to_gradio_chatbot(), results['error_msg'])
    return (dialog_state, input_state, chatbot) + (enable_btn, ) * 4


def load_demo(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    dialog_state = conv_seed_llama.copy()
    input_state = init_input_state()
    return dialog_state, input_state


title = ("""
# SEED-LLaMA
[[Project Page]](https://ailab-cvc.github.io/seed/seed_llama.html) [[Paper]](https://arxiv.org/pdf/2310.01218.pdf) [[Code]](https://github.com/AILab-CVC/SEED/tree/main)

## Tips:
* Check out the conversation examples (at the bottom) for inspiration.

* You can adjust "Max History Rounds" to try a conversation with up to five rounds. For more turns, you can download our checkpoints from GitHub and deploy them locally for inference.

* Our demo supports a mix of images and texts as input. You can freely upload an image or enter text, and then click on "Add Image/Text". You can repeat the former step multiple times, and click on "Submit" for model inference at last.

* If you are not satisfied with the output, especially the generated image, you may click on "Regenerate" for another chance.

* You can click "Force Image Generation" to compel the model to produce images when necessary. For example, our model might struggle to generate images when there is an excessive amount of text-only context.
* SEED-LLaMA was trained with English-only data. It may process with other languages due to the inherent capabilities from LLaMA, but might not stable.
""")

css = """
img {
  font-family: 'Helvetica';
  font-weight: 300;
  line-height: 2;  
  text-align: center;
  
  width: auto;
  height: auto;
  display: block;
  position: relative;
}

img:before { 
  content: " ";
  display: block;

  position: absolute;
  top: -10px;
  left: 0;
  height: calc(100% + 10px);
  width: 100%;
  background-color: rgb(230, 230, 230);
  border: 2px dotted rgb(200, 200, 200);
  border-radius: 5px;
}

img:after { 
  content: " ";
  display: block;
  font-size: 16px;
  font-style: normal;
  font-family: FontAwesome;
  color: rgb(100, 100, 100);
  
  position: absolute;
  top: 5px;
  left: 0;
  width: 100%;
  text-align: center;
}

"""

if __name__ == '__main__':

    examples_mix = [
        ['images/cat.jpg', 'Add sunglasses to the animal.'],
        ['images/eagle.jpg', 'Transform this image into cartoon style'],
        [None, 'Generate an image of dog on green grass.'],
        [None, 'Draw a painting of sunflowers in Van Gogh style.'],
        ['images/dogs_4.jpg', 'How many dogs in the image?'],
        ['images/spongebob.png', 'Who are they?'],
        ['images/star.jpg', 'Do you know this painting?'],
    ]
    
    examples_conv = [
        ['images/demo_example1.jpg'],
        ['images/demo_example2.jpg'],
        ['images/demo_example3.jpg'],
        ['images/demo_example7.jpg'],
        ['images/demo_example5.jpg'],
        ['images/demo_example6.jpg'],
    ]
    
    with gr.Blocks(css=css) as demo:
        gr.Markdown(title)
        dialog_state = gr.State()
        input_state = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    image = gr.Image(type='pil', label='input_image')
                with gr.Row():
                    text = gr.Textbox(lines=5,
                                      show_label=False,
                                      label='input_text',
                                      elem_id='textbox',
                                      placeholder="Enter text or add image, and press submit,").style(container=False)
                with gr.Row():
                    add_image_btn = gr.Button("Add Image")
                    add_text_btn = gr.Button("Add Text")

                    submit_btn = gr.Button("Submit")

                with gr.Row():
                    num_beams = gr.Slider(minimum=1, maximum=4, value=1, step=1, interactive=True, label="Num of Beams")
                    max_new_tokens = gr.Slider(minimum=64,
                                               maximum=1024,
                                               value=256,
                                               step=64,
                                               interactive=True,
                                               label="Max New Tokens")
                    temperature = gr.Slider(minimum=0.0,
                                            maximum=1.0,
                                            value=1.0,
                                            step=0.1,
                                            interactive=True,
                                            label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, interactive=True, label="Top P")
                    max_turns = gr.Slider(minimum=1, maximum=5, value=3, step=1, interactive=True, label="Max History Rounds")
                    force_img_gen = gr.Radio(choices=[True, False], value=False, label='Force Image Generation')

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id='chatbot', label="SEED LLaMA").style(height=700)
                with gr.Row():
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        # with gr.Row():
        #     gr.Examples(examples=examples_image, label='Image examples', inputs=[image])
        with gr.Row():
            # with gr.Column(scale=6):
            gr.Examples(examples=examples_mix, label='Input examples', inputs=[image, text])
            # with gr.Column(scale=0.4):
            #     gr.Examples(examples=examples_text, inputs=[text])
            
        
        # with gr.Row():
        #     gr.Examples(examples=examples_2, inputs=[image])
        
        with gr.Row():
            # gr.Gallery(value=[Image.open(e[0]) for e in examples_conv], show_label=True, label="Example Conversations", elem_id="gallery",height=1400, object_fit='contain').style(grid=[3], height='auto')
            gr.Gallery(value=[Image.open(e[0]) for e in examples_conv], show_label=True, label="Example Conversations", elem_id="gallery",height=1500, columns=[3], rows=[2])
        
        # Register listeners
        btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response, [dialog_state], [upvote_btn, downvote_btn])
        downvote_btn.click(downvote_last_response, [dialog_state], [upvote_btn, downvote_btn])
        regenerate_btn.click(regenerate, [dialog_state], [dialog_state, chatbot] + btn_list).then(
            http_bot, [dialog_state, input_state, temperature, top_p, max_new_tokens, num_beams, max_turns, force_img_gen],
            [dialog_state, input_state, chatbot] + btn_list)
        add_image_btn.click(add_image, [dialog_state, input_state, image],
                            [dialog_state, input_state, image, chatbot] + btn_list)

        add_text_btn.click(add_text, [dialog_state, input_state, text], [dialog_state, input_state, text, chatbot] + btn_list)

        submit_btn.click(
            add_image, [dialog_state, input_state, image], [dialog_state, input_state, image, chatbot] + btn_list).then(
                add_text, [dialog_state, input_state, text],
                [dialog_state, input_state, text, chatbot, upvote_btn, downvote_btn, regenerate_btn, clear_btn]).then(
                    http_bot, [dialog_state, input_state, temperature, top_p, max_new_tokens, num_beams, max_turns, force_img_gen],
                    [dialog_state, input_state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [dialog_state, input_state, chatbot] + btn_list)
        
        demo.load(load_demo, None, [dialog_state, input_state])

    demo.launch(server_name=args.server_name, server_port=args.server_port, enable_queue=True)
