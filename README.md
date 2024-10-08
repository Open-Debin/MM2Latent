# MM2Latent: Text-to-facial image generation and editing in GANs with multimodal assistance
[![arXiv](https://img.shields.io/badge/arXiv-2409.11010-b31b1b.svg)](https://www.arxiv.org/abs/2409.11010)
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://www.arxiv.org/abs/2409.11010)

Authors' official PyTorch implementation of the ***"MM2Latent: Text-to-facial image generation and editing in GANs with multimodal assistance"***, accepted in the Advances in Image Manipulation workshop (**AIM**) Workshop of **ECCV 2024**. If you find this code useful for your research, please [cite](#citation) our paper.

> [**MM2Latent: Text-to-facial image generation and editing in GANs with multimodal assistance"**] <br>Debin Meng, Christos Tzelepis, Ioannis Patras, and Georgios Tzimiropoulos<br>
> Advances in Image Manipulation (AIM) Workshop of ECCV 2024.<br>
> **Abstract:** Generating human portraits is a hot topic in the image generation area, e.g. mask-to-face generation and text-to-face generation. However, these unimodal generation methods lack controllability in image generation. Controllability can be enhanced by exploring the advantages and complementarities of various modalities. For instance, we can utilize the advantages of text in controlling diverse attributes and masks in controlling spatial locations. Current state-of-the-art methods in multimodal generation face limitations due to their reliance on extensive hyperparameters, manual operations during the inference stage, substantial computational demands during training and inference, or inability to edit real images. In this paper, we propose a practical framework — MM2Latent — for multimodal image generation and editing. We use StyleGAN2 as our image generator, FaRL for text encoding, and train an autoencoders for spatial modalities like mask, sketch and 3DMM. We propose a strategy that involves training a mapping network to map the multimodal input into the w latent space of StyleGAN. The proposed framework 1) eliminates hyperparameters and manual operations in the inference stage, 2) ensures fast inference speeds, and 3) enables the editing of real images. Extensive experiments demonstrate that our method exhibits superior performance in multimodal image generation, surpassing recent GAN- and diffusion-based methods. Also, it proves effective in multimodal image editing and is faster than GAN- and diffusion-based methods. 
![alt text](image/teaser.jpg)



**Code coming soon...**



## Citation

If you find this work useful, please consider citing it:
```
@article{meng2024mm2latent,
  title={MM2Latent: Text-to-facial image generation and editing in GANs with multimodal assistance},
  author={Meng, Debin and Tzelepis, Christos and Patras, Ioannis and Tzimiropoulos, Georgios},
  journal={arXiv preprint arXiv:2409.11010},
  year={2024}
}
```


## Acknowledgment

This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.
