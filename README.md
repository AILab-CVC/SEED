# :chestnut: SEED Multimodal

[![Project Homepage](https://img.shields.io/badge/Project-Homepage-green)](https://ailab-cvc.github.io/seed/)
[![arXiv](https://img.shields.io/badge/arXiv-2307.08041-b31b1b.svg)](https://arxiv.org/abs/2307.08041)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01218-b31b1b.svg)](https://arxiv.org/abs/2310.01218)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-orange)](https://huggingface.co/AILab-CVC/SEED/tree/main)
[![Demo](https://img.shields.io/badge/Gradio-Demo-orange)](https://10a4e7976e6fc2032c.gradio.live/)


**Powered by [CV Center, Tencent AI Lab](https://ailab-cvc.github.io), and [ARC Lab, Tencent PCG](https://github.com/TencentARC).**

![image](https://github.com/AILab-CVC/SEED/blob/main/paper_images/milestone.jpg)

The repository provides the official implementation of [SEED](https://ailab-cvc.github.io/seed/seed.html), [SEED-LLaMA](https://ailab-cvc.github.io/seed/seed_llama.html). For any inquiries, please email [seed-x@googlegroups.com](mailto:seed-x@googlegroups.com).


## News

**:beers: We are actively looking for self-motivated interns. Please feel free to reach out if you are interested. :beers:**

- [x] **2023-10-20** : We release the [checkpoints](https://huggingface.co/AILab-CVC/SEED/tree/main) and code of the SEED-2 tokenizer, and SEED-LLaMA-8B/14B. 
- [x] **2023-10-20** : We release [online demo for SEED-LLaMA](https://10a4e7976e6fc2032c.gradio.live/), please feel free to use it by yourself.
- [x] **2023-10-02** :paperclip: We release the technical report of SEED-LLaMA on arXiv, which is empowered by the improved SEED-2 tokenizer.
- [x] **2023-07-29** :octocat: We release the checkpoint of the SEED tokenizer and its inference code. [[Getting started]](#seed-1-tokenizer)
- [x] **2023-07-16** :paperclip: We release the technical report of SEED on arXiv.

Stay tuned for the updates!

## Brief Introduction

It is recommended to check out our [papers](#citation) for technical details.

### :speech_balloon: What can SEED-LLaMA do?

![image](https://github.com/AILab-CVC/SEED/blob/main/paper_images/v2/teaser.jpg)

**SEED-LLaMA** is capable of both multimodal comprehension and generation, exhibiting compositional emergent abilities such as multi-turn in-context multimodal generation, acting like your AI assistant. [[Compare to SOTA]](https://ailab-cvc.github.io/seed/seed_llama_compare.html) [[More examples on X]](https://twitter.com/ge_yixiao/status/1710509538238157069?s=20)

<!-- We present **SEED-LLaMA** by large-scale pretraining and instruction tuning on the interleaved textual and visual data, which demonstrates impressive performance on a broad range of multimodal comprehension and generation tasks. More importantly, SEED-LLaMA has exhibited **compositional emergent abilities** such as multi-turn in-context multimodal generation, acting like your **AI assistant**. -->

### :bulb: How does SEED-LLaMA achieve it?

![image](https://github.com/AILab-CVC/SEED/blob/main/paper_images/seed_overview.jpg)

The core of SEED-LLaMA is the tailored **SEED** tokenizer, which properly quantized visual signals into discrete visual tokens, capturing necessary semantics while being produced under 1D causal dependence. [[SEED-2 vs. SEED-1]](https://ailab-cvc.github.io/seed/seed_llama.html)

<!-- ### Compositional Emergent Ability
**Multi-turn in-context image and text generation.**
![image](paper_images/v2/multi_turn1.jpg)
![image](paper_images/v2/multi_turn2.jpg)

**Compositional image generation.**
![image](paper_images/v2/results.jpg) -->

<!-- ### SEED Tokenizer v2
In SEED tokenizer v2, the generation embedding is aligned with the **image embedding** (1 token) of [unCLIP SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip), and can be decoded to realistic images with the unCLIP-SD-UNet. In SEED tokenizer v1, we train a visual tokenizer through aligning the **generation embeddings** with the text embeddings (77 tokens) of [SD](https://github.com/CompVis/stable-diffusion), and the generation embeddings can be decoded to images with the SD-UNet. The below figure shows the visual comparison of the reconstructed images between SEED tokenizer v2 (the third row) and SEED tokenizer v1 (the second row). We can observe that the images reconstructed by SEED tokenizer v2 can better preserve the visual information of the original images. The semantic representations of texts can not fully preserve the rich visual information of images.
![image](paper_images/v2/seed_comparison.jpg) -->

<!-- ### Pretraining
We perform multimodal autoregressive pretraining on interleaved visual and textual data for SEED-LLaMA. Visual inputs are pre-processed into discrete tokens to conserve computational resources. Given the multimodal discrete sequence, a unified next-word-prediction objective is employed. During inference, visual codes are decoded into a realistic image by SEED De-Tokenization.
![image](paper_images/v2/method_page.jpg) -->

## Usage

### Dependencies
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.11.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation
Clone the repo and install dependent packages

  ```bash
  git clone https://github.com/AILab-CVC/SEED.git
  cd SEED
  pip install -r requirements.txt
  ```

    
### Model Weights
We release the pretrained SEED Tokenizer and De-Tokenizer, instruction tuned SEED-LLaMA-8B and SEED-LLaMA-14B in [SEED Hugging Face](https://huggingface.co/AILab-CVC/SEED). 
Please download the checkpoints and save under the folder `./pretrained`.

```bash
cd pretrained   # SEED/pretrained
git lfs install
git clone https://huggingface.co/AILab-CVC/SEED
mv SEED/* ./
```

To reconstruct the image from the SEED visual codes using unCLIP SD-UNet, please download the pretrained [unCLIP SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip).
Rename the checkpoint directory to **"diffusion_model"** and create a soft link to the "pretrained/seed_tokenizer" directory.

```bash
# SEED/pretrained
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip
mv stable-diffusion-2-1-unclip seed_tokenizer/diffusion_model
```


### Inference for visual tokenization and de-tokenization
To discretize an image to 1D visual codes with causal dependency, and reconstruct the image from the visual codes using the off-the-shelf unCLIP SD-UNet:

```bash
cd ..   # SEED/ 
python scripts/seed_tokenizer_inference.py
```

### Launching Gradio Demo of SEED-LLaMA-14B Locally 
Building the local demo of SEED-LLaMA-14B currently requires 2*32GB devices.

```bash
# SEED/
# in first terminal
sh scripts/start_backend.sh
# in second terminal
sh scripts/start_frontend.sh
``` 
Then the demo can be accessed through http://127.0.0.1:80

## Citation
If you find the work helpful, please consider citing:
```bash
@article{ge2023making,
  title={Making LLaMA SEE and Draw with SEED Tokenizer},
  author={Ge, Yuying and Zhao, Sijie and Zeng, Ziyun and Ge, Yixiao and Li, Chen and Wang, Xintao and Shan, Ying},
  journal={arXiv preprint arXiv:2310.01218},
  year={2023}
}

@article{ge2023planting,
  title={Planting a seed of vision in large language model},
  author={Ge, Yuying and Ge, Yixiao and Zeng, Ziyun and Wang, Xintao and Shan, Ying},
  journal={arXiv preprint arXiv:2307.08041},
  year={2023}
}
```

The project is still in progress. Stay tuned for more updates!

## License
`SEED` is released under [Apache License Version 2.0](License.txt). 

`SEED-LLaMA` is released under the original [License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) of [LLaMA2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).

## Acknowledgement
We thank the great work from [unCLIP SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip) and [BLIP2](https://github.com/salesforce/LAVIS).

