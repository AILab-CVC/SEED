# :chestnut: SEED Multimodal

[![Project Homepage](https://img.shields.io/badge/Project-Homepage-green)](https://ailab-cvc.github.io/seed/)
[![arXiv](https://img.shields.io/badge/arXiv-2307.08041-b31b1b.svg)](https://arxiv.org/abs/2307.08041)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01218-b31b1b.svg)](https://arxiv.org/abs/2310.01218)

**Powered by [CV Center, Tencent AI Lab](https://ailab-cvc.github.io), and [ARC Lab, Tencent PCG](https://github.com/TencentARC).**

![image](paper_images/milestone.jpg)

The repository provides the official implementation of [SEED](https://ailab-cvc.github.io/seed/seed.html), [SEED-LLaMA](https://ailab-cvc.github.io/seed/seed_llama.html). For any inquiries, please email [seed-x@googlegroups.com](mailto:seed-x@googlegroups.com).


## News

**:beers: We are actively looking for self-motivated interns. Please feel free to reach out if you are interested. :beers:**

- [ ] :eyes: Release the checkpoints and code of the SEED-2 tokenizer, and SEED-LLaMA-8B/14B. Expected to be in late October.
- [ ] :eyes: We will soon release an online demo for SEED-LLaMA.
- [x] **2023-10-02** :paperclip: We release the technical report of SEED-LLaMA on arXiv, which is empowered by the improved SEED-2 tokenizer.
- [x] **2023-07-29** :octocat: We release the checkpoint of the SEED tokenizer and its inference code. [[Getting started]](#seed-1-tokenizer)
- [x] **2023-07-16** :paperclip: We release the technical report of SEED on arXiv.

Stay tuned for the updates!

## SEED Tokenizer v1
[[arXiv]](https://arxiv.org/abs/2307.08041)

![image](paper_images/teaser.jpg)

## SEED Tokenizer v1 for Image Reconstruction
![image](paper_images/reconstruction.jpg)

## SEED-OPT<sub>2.7B </sub> for Multimodal Comprehension
![image](paper_images/vqa.jpg)

## SEED-OPT<sub>2.7B </sub> for Multimodal Generation
![image](paper_images/generation.jpg)

## Dependencies and Installation
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.11.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo

    ```bash
    git clone https://github.com/AILab-CVC/SEED.git
    cd SEED
    ```

2. Install dependent packages

    ```bash
    sh install.sh
    ```

## Model Weights
We release the pre-trained SEED Visual Tokenizer in [google drive](https://drive.google.com/drive/folders/1xmVXuttQfBPBOe4ZR96Wu1X34uzPkxsS?usp=drive_link).

## Inference
To discretize an image to 1D vision codes with causal dependency, and reconstruct the image
from the vision codes using stable diffusion UNet,

1. Download the pre-trained SEED Visual Tokenizer and stable diffusion model in [google drive](https://drive.google.com/drive/folders/1xmVXuttQfBPBOe4ZR96Wu1X34uzPkxsS?usp=drive_link) and put them under the folder "pretrained".
2. run the inference code.
```bash
    python demo_recon.py
  ```

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

## Acknowledgement
We utilize `Stable Diffusion` to decode images from our visual codes, and use its implementation and pre-trained model in https://github.com/CompVis/stable-diffusion. Our code is developped based on https://github.com/salesforce/LAVIS. Thanks for their wonderful works.
