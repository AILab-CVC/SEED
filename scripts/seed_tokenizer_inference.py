import hydra
from omegaconf import OmegaConf
from PIL import Image
import pyrootutils
import os

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
transform_cfg_path = 'configs/transform/clip_transform.yaml'

image_path = 'images/cat.jpg'
save_dir = './'
save_path = os.path.join(save_dir, os.path.basename(image_path))

os.makedirs(save_dir, exist_ok=True)

device = 'cuda'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

image = Image.open(image_path).convert('RGB')

image_tensor = transform(image).to(device)
image_ids = tokenizer.encode_image(image_torch=image_tensor)

images = tokenizer.decode_image(image_ids)

images[0].save(save_path)