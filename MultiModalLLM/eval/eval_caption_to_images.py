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
image_id_shift = 32000
save_dir = 'generated_images'

model_cfg = OmegaConf.load(model_cfg_path)
tokenizer_cfg = OmegaConf.load(tokneizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)
model = hydra.utils.instantiate(model_cfg, tokenizer=tokenizer, model_id=ckpt_path).eval().half().to(device)

print('Init Done')
generation_config = transformers.GenerationConfig(temperature=0.0, num_beams=4)
guidance_scale = 5.0

os.makedirs(save_dir, exist_ok=True)

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
        generation_config=generation_config,
        max_new_tokens=120,
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]

    return generate_ids


def generate_image(tokenizer, generate_ids, save_path, shift, negative_ids=None, guidance_scale=10):
    eoi_index = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0][0]
    generate_ids = (generate_ids[:eoi_index] - shift).reshape(1, -1)
    print(generate_ids)
    images = tokenizer.decode_image(generate_ids, negative_ids, guidance_scale)
    images[0].save(save_path)


caption_all = [
    'A Pikachu fine dining with a view to the Eiffel Tower',
    'A small cabin on top of a snowy mountain in the style of Disney, artstation',
    'kingfisher bird, by Akihito Yoshida, Cool Color Palette',
    'Elon Musk on SpaceX, Cool Color Palette',
    'Elon Musk on Tesla Giga factory, realistic',
    'whale jumping out of water, oil painting, 4k',
    'Rabbit Hobbit, standing proudly in the shire',
    'An airport with modern landscape architectural design for industrialpunk',
    'A majestic colorful dragon, ultra realistic',
    'A mysterious goblin town, realistic',
    'A monochrome forest of ebony trees, octane rendered',
    'Cute little rabbit eating carrot in the forest, highly detailed',
    'Taj Mahal in colorful light, oil painting by Claude Monet',
    'A massive city of 300000 people with a golden citadel one of the most breathtaking castles in the world',
    'Tiny cute and adorable piglet adventurer dressed in a warm overcoat with survival gear on a winters day, jean - baptiste monge , anthropomorphic',
    '1950s retro science fiction rat rod race car. muted colors. by jean - baptiste monge',
    'portrait of young nordic girl, age 25, freckled skin, neck tatoo, blue eyes, blond hair, 35mm lens, photography, ultra details',
    'kneeling cat knight, portrait, finely detailed armor, intricate design, silver, silk, cinematic lighting, 4k',
    'two flamingos are in love in a sunset, concept art. high quality',
    'a rose bush with many vivid colored roses, weta digital, scrupulous detail, excessive detail, highly real, dreamlike lighting',
    'astronaut with jetpack in space, lens flares, stars, planets, celestial, full length portrait, artistic portrait, rigorous detail, precise detail, god rays',
    'An illustration of a red owl with bright blue eyes.',
    'A group of explorers traversing a lush, alien planet filled with strange flora and fauna,',
    'An underwater city teeming with all manner of exotic marine life and advanced technology,',
    'A massive space station orbiting a distant, hostile planet, serving as a hub for interstellar trade and diplomacy',
    'A lone wanderer traversing a desolate wasteland, armed only with their wits and whatever supplies they can scavenge',
    'A cybernetic soldier, enhanced with advanced weapons systems and tactical analysis software, on a mission behind enemy lines',
    'A detective, enhanced with advanced forensic analysis tools and facial recognition software, trying to solve a series of cyber-crimes in a futuristic city,',
    'A human-robot hybrid, struggling to find their place in a world that is divided between organic and synthetic life forms',
    'A space explorer, equipped with a suite of cybernetic enhancements designed for long-duration space travel, on a mission to explore the furthest reaches of the galaxy,',
    'A planet where the atmosphere is made of a dense, swirling mist that obscures all vision',
    'A steampunk city filled with airships and clockwork automatons, powered by steam and coal',
    'A beautiful humanoid with a rose in her hand',
    'A celestial entity floating serenely in an endless void, cradling a shattered star as a symbol of hope and rebirth.',
    'A robotic entity with a kaleidoscopic body, constantly shifting and refracting light.',
    'A cyborg mercenary with a bionic eye and enhanced reflexes, feared and revered in equal measure.',
    'A sorceress with cybernetic enhancements, wielding a staff that crackles with arcane energy.',
    'A guardian spirit trapped within a robotic exoskeleton, sworn to protect the balance of nature.',
    'A celestial deity manifesting as a colossal robotic titan, overseeing the cosmic order.',
    'A bioengineered creature, merging plant and animal DNA to create a harmonious hybrid form.',
    'n ancient golem awakened in a modern city, grappling with its purpose in the digital age.',
    'Draw a cute kitten with big eyes and a fluffy tail.',
    'Draw a simple flower with petals, stem, and leaves.',
    'A dramatic sunset over a vast desert landscape.',
    'A colorful and vibrant fireworks display lighting up the night sky.',
    'A soaring hot air balloon floating over a scenic landscape.',
    'A playful kitten chasing a butterfly in a meadow.',
    'An underwater city populated by fantastical sea creatures.',
    'A whimsical treehouse village hidden in a dense forest.',
    'A bustling street market in a vibrant foreign city.',
    'A serene winter landscape with snow-covered trees and a cozy cabin.',
    'A majestic waterfall cascading down a rocky cliff.',
    'A magical portal to another world, hidden in a mysterious cave.',
    'A group of fairies frolicking in a enchanted glen.',
    'An ancient castle perched atop a misty mountain.',
    'A bustling harbor scene with boats, docks, and seagulls.',
    'A serene Japanese garden with cherry blossoms in full bloom.',
    'A futuristic cityscape with skyscrapers and flying car',
    'A peaceful countryside scene with rolling hills and grazing animals.',
    'A dramatic sunset over a vast desert landscape.',
    'A cozy cafe with steaming mugs of coffee and delectable pastries.',
    'A colorful and vibrant fireworks display lighting up the night sky.',
    'A tranquil beach scene with crashing waves and a sandy shore.',
    'A mystical forest with glowing mushrooms and enchanted creatures.',
    'A bustling carnival with rides, games, and laughter.',
    'A serene lake scene with boats and reflections of the surrounding nature.',
    'A lush and tropical jungle teeming with exotic wildlife.',
    'A whimsical circus with acrobats, clowns, and circus animals.',
    'A futuristic space station with spacecraft and distant planets in the background.',
    'A serene countryside scene with a rustic farmhouse and grazing animals.',
    'A bustling city street at rush hour, filled with cars, buses, and pedestrians.',
    'A serene autumn landscape with colorful foliage and a peaceful pond.',
    'A dreamy and surreal landscape with floating islands and unusual features.',
    'A cozy and inviting library filled with books and reading nooks.',
    'A lively street market with street performers, food stalls, and colorful decorations.',
    'A dramatic stormy seascape with crashing waves and dark clouds.',
    'A whimsical underwater tea party with sea creatures and mermaids.',
    'A serene and peaceful meditation garden with flowing water and Zen elements.',
    'A bustling airport scene with planes, baggage carts, and travelers.',
    'a cupcake with frosting, sprinkles, and a cherry on top.',
    'a beautiful rainbow with fluffy clouds and a pot of gold at the end.',
    'a farm scene with barns, tractors, and farm animals.',
    'a cute kitten with big eyes and a fluffy tail.',
    'a cupcake with frosting, sprinkles, and a cherry on top.',
    'a couple on a beach, walking hand-in-hand, with the sun setting over the horizon and waves crashing at their feet.',
    'a couple enjoying a bike ride together, with the wind in their hair and smiles on their faces.',
]

with torch.no_grad():
    input_tokens = tokenizer.bos_token + BOI_TOKEN
    negative_ids = generate(model=model, tokenizer=tokenizer, input_tokens=input_tokens, generation_config=generation_config)
    eoi_index = torch.where(negative_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0][0]
    negative_ids = (negative_ids[:eoi_index] - image_id_shift).reshape(1, -1)
    print('negative_ids: ', negative_ids)
    # negative_ids = None

    for caption in caption_all:
        save_path = os.path.join(save_dir, caption + '_.jpg')

        input_tokens = tokenizer.bos_token + caption + BOI_TOKEN

        generate_ids = generate(model=model,
                                tokenizer=tokenizer,
                                input_tokens=input_tokens,
                                generation_config=generation_config)

        generate_image(tokenizer=tokenizer,
                       generate_ids=generate_ids,
                       save_path=save_path,
                       shift=image_id_shift,
                       negative_ids=negative_ids,
                       guidance_scale=guidance_scale)
