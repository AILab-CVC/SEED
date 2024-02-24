import hydra
from omegaconf import OmegaConf
import pyrootutils
import torch
import numpy as np
import random
import transformers
import os

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

model_cfg_path = 'configs/model/vicuna_7b_lora_pretrained.yaml'
tokneizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer.yaml'
ckpt_path = 'log/seed_vicuna-7b_lora_pretrain/checkpoint-10000'
device = 'cuda:0'

model_cfg = OmegaConf.load(model_cfg_path)
tokenizer_cfg = OmegaConf.load(tokneizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)
model = hydra.utils.instantiate(model_cfg, tokenizer=tokenizer, model_id=ckpt_path).eval().half().to(device)

print('Init Done')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)


def generate(model, tokenizer, input_tokens, generation_config):

    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to(device)

    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]

    return generate_ids

def get_transform(name, image_size):
    if name == 'vqkd':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    elif name == 'blip':
        transform = transforms.Compose([
            # transforms.Resize((image_size, image_size), interpolation=3),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        raise NotImplementedError

    return transform

transform = get_transform('blip', 224)
generation_config = {
        'temperature': 1.0,
        'num_beams': 5,
        'max_new_tokens': 512,
        'top_p': 0.5,
        'do_sample': True}

with torch.no_grad():
    image_path = 'demo_image.jpg'
    question = 'Question: what is the woman doing?\nAnswer:'

    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cuda")
    img_ids = tokenizer.encode_image(image_torch=image)
    img_ids = img_ids.view(-1).cpu().numpy()
    img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids]) + EOI_TOKEN

    input_tokens = tokenizer.bos_token + img_tokens + question

    generate_ids = generate(model=model,
                            tokenizer=tokenizer,
                            input_tokens=input_tokens,
                            generation_config=generation_config)

    generate_text = tokenizer.decode(generate_ids, skip_special_tokens=True)
    print(generate_text)