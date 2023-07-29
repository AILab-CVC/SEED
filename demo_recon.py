import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import logging
logging.set_verbosity_error()

from models.visual_tokenizer import SEEDVisualTokenizer
from stable_diffusion.scripts.txt2img import generate

# image preprocessing
def get_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    return transform


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# load sample image
image_paths = ['demos/cat.jpg']
for image_path in image_paths:
    raw_image = Image.open(image_path).convert("RGB")
    image = get_transform(image_size=224)(raw_image).unsqueeze(0).to(device)

    # SEED Tokenizer
    SEED_tokenizer = SEEDVisualTokenizer().to(device)

    with torch.no_grad():
        # visual_codes: 1D discrete vision codes with causal dependency
        visual_codes, reverse_output_up = SEED_tokenizer.predict(image)
        print(visual_codes)
        
        # Reconstruct images with SD U-Net
        generate(text_features=reverse_output_up)
