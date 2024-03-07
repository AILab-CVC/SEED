import torch
from PIL import Image
import numpy as np
import random
from diffusers import StableUnCLIPImg2ImgPipeline
import os


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)


shape_latents = torch.Size([1, 4, 96, 96])
latents = torch.randn(shape_latents, generator=None, device="cuda", dtype=torch.float32, layout=torch.strided).to("cuda")

shape_noise = torch.Size([1, 1024])
noise = torch.randn(shape_noise, generator=None, device="cuda", dtype=torch.float32, layout=torch.strided).to("cuda")

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32, variation="fp16"
)
pipe = pipe.to("cuda")


from lavis.models import load_model_and_preprocess

model, vis_processors, _ = load_model_and_preprocess(name="blip2_codebook_all_image", model_type="pretrain", is_eval=True, device=device)
model = model.eval()

image_path = 'demo_images/dog3.jpg'
raw_image = Image.open(image_path).convert("RGB")

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
with torch.no_grad():
    reverse_output_up = model.get_discrete_features_for_decode(image)

images = pipe(image_embeds=reverse_output_up, noise_level=0, num_inference_steps=20, latents=latents, noise=noise).images

images[0].save('recon_dog.png')


