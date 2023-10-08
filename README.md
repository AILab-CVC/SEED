# Planting a SEED of Vision in Large Language Model


## News
**2023-09-27** SEED-LLaMA will be released in October. Stay tuned for the updates!

**2023-09-09** We are actively looking for self-motivated interns. Please feel free to reach out if you are interested.

**2023-07-29** We release the pre-trained SEED Visual Tokenizer and inference code.

## SEED-LLaMA (Making LLaMA SEE and Draw with SEED Tokenizer)
[[arXiv]](https://arxiv.org/abs/2310.01218)

![image](paper_images/v2/teaser.jpg)

We present **SEED-LLaMA** by large-scale pretraining and instruction tuning on the interleaved textual and visual data, which demonstrates impressive performance on a broad range of multimodal comprehension and generation tasks. More importantly, SEED-LLaMA has exhibited **compositional emergent abilities** such as multi-turn in-context multimodal generation, acting like your **AI assistant**.

### Compositional Emergent Ability
**Multi-turn in-context image and text generation.**
![image](paper_images/v2/multi_turn1.jpg)
![image](paper_images/v2/multi_turn2.jpg)

**Compositional image generation.**
![image](paper_images/v2/results.jpg)

### SEED Tokenizer v2
In SEED tokenizer v2, the generation embedding is aligned with the **image embedding** (1 token) of [unCLIP SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip), and can be decoded to realistic images with the unCLIP-SD-UNet. In SEED tokenizer v1, we train a visual tokenizer through aligning the **generation embeddings** with the text embeddings (77 tokens) of [SD](https://github.com/CompVis/stable-diffusion), and the generation embeddings can be decoded to images with the SD-UNet. The below figure shows the visual comparison of the reconstructed images between SEED tokenizer v2 (the third row) and SEED tokenizer v1 (the second row). We can observe that the images reconstructed by SEED tokenizer v2 can better preserve the visual information of the original images. The semantic representations of texts can not fully preserve the rich visual information of images.
![image](paper_images/v2/seed_comparison.jpg)

### Pretraining
We perform multimodal autoregressive pretraining on interleaved visual and textual data for SEED-LLaMA. Visual inputs are pre-processed into discrete tokens to conserve computational resources. Given the multimodal discrete sequence, a unified next-word-prediction objective is employed. During inference, visual codes are decoded into a realistic image by SEED De-Tokenization.
![image](paper_images/v2/method_page.jpg) 

## [SEED Tokenizer v1](https://github.com/AILab-CVC/SEED/blob/main/SEED%20Tokenizer%20v1.md)

## To Do
- [ ] Release SEED Tokenizer v2
- [ ] Release SEED-LLaMA demo

## Citation
If you find the work helpful, please consider citing:
```
@article{ge2023making,
  title={Making LLaMA SEE and Draw with SEED Tokenizer},
  author={Ge, Yuying and Zhao, Sijie and Zeng, Ziyun and Ge, Yixiao and Li, Chen and Wang, Xintao and Shan, Ying},
  journal={arXiv preprint arXiv:2310.01218},
  year={2023}
}
@misc{ge2023planting,
      title={Planting a SEED of Vision in Large Language Model}, 
      author={Yuying Ge and Yixiao Ge and Ziyun Zeng and Xintao Wang and Ying Shan},
      year={2023},
      eprint={2307.08041},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

The project is still in progress. Stay tuned for more updates!
