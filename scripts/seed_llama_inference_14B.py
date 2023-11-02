import hydra

import pyrootutils
import os
import torch

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000


def generate(tokenizer, input_tokens, generation_config, model):

    input_ids = tokenizer(
        input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to("cuda")

    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]

    return generate_ids


def decode_image_text(generate_ids, tokenizer, save_path=None):

    boi_list = torch.where(generate_ids == tokenizer(
        BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(
        EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        print(texts)

    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]

        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            print(texts)

        image_ids = (generate_ids[boi_index+1:eoi_index] -
                     image_id_shift).reshape(1, -1)

        images = tokenizer.decode_image(image_ids)

        images[0].save(save_path)


device = "cuda"

tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(
    tokenizer_cfg, device=device, load_diffusion=True)

transform_cfg_path = 'configs/transform/clip_transform.yaml'
transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

model_cfg = OmegaConf.load('configs/llm/seed_llama_14b.yaml')
model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
model = model.eval().to(device)

generation_config = {
    'temperature': 1.0,
    'num_beams': 1,
    'max_new_tokens': 512,
    'top_p': 0.5,
    'do_sample': True
}

s_token = "[INST] "
e_token = " [/INST]"
sep = "\n"


# visual question answering
image_path = "images/cat.jpg"
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).to(device)
img_ids = tokenizer.encode_image(image_torch=image_tensor)
img_ids = img_ids.view(-1).cpu().numpy()
img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item)
                                 for item in img_ids]) + EOI_TOKEN

question = "What is this animal?"

input_tokens = tokenizer.bos_token + s_token + \
    img_tokens + question + e_token + sep
generate_ids = generate(tokenizer, input_tokens, generation_config, model)
decode_image_text(generate_ids, tokenizer)

# text-to-image generation
prompt = "Can you generate an image of a dog on the green grass?"
input_tokens = tokenizer.bos_token + s_token + prompt + e_token + sep
generate_ids = generate(tokenizer, input_tokens, generation_config, model)
save_path = 'dog.jpg'
decode_image_text(generate_ids, tokenizer, save_path)

# multimodal prompt image generation
instruction = "Can you make the cat wear sunglasses?"
input_tokens = tokenizer.bos_token + s_token + \
    img_tokens + instruction + e_token + sep
generate_ids = generate(tokenizer, input_tokens, generation_config, model)
save_path = 'cat_sunglasses.jpg'
decode_image_text(generate_ids, tokenizer, save_path)
