# GaussianGPT Checkpoints

Pre-trained VQ-VAE and GPT checkpoints. Each GPT must be paired with the VQ-VAE listed alongside it. Please report download or checkpoint issues on the [GaussianGPT GitHub](https://github.com/nicolasvonluetzow/GaussianGPT).

## Trained on 3D-FRONT

```sh
wget https://kaldir.vc.cit.tum.de/gaussiangpt/vqvae_vfront.ckpt  # ~2.0 GB
wget https://kaldir.vc.cit.tum.de/gaussiangpt/gpt_vfront.ckpt    # ~3.2 GB
```

## Pre-trained on 3D-FRONT + ASE, fine-tuned on 3D-FRONT

```sh
wget https://kaldir.vc.cit.tum.de/gaussiangpt/vqvae_both.ckpt  # ~2.0 GB
wget https://kaldir.vc.cit.tum.de/gaussiangpt/gpt_both.ckpt    # ~3.2 GB
```

## Trained on PhotoShape (object-level)

```sh
wget https://kaldir.vc.cit.tum.de/gaussiangpt/vqvae_photoshape.ckpt  # ~1.2 GB
wget https://kaldir.vc.cit.tum.de/gaussiangpt/gpt_photoshape.ckpt    # ~1.7 GB
```

## SHA256

```
9f70d0939dc791292be52da6c503bf51b3ac73d9905b51784d0aac81e44faf7a  vqvae_vfront.ckpt
203dc730495bf4f21e60280c6152703867f1b81e9c035d110d792e6a87d9313b  gpt_vfront.ckpt
a780ed2920e877736699bb3da84a43362c1a565adff91217841d4b95cb784542  vqvae_both.ckpt
054978e811716292472dd501cc05b54c39789af6dd8730aab86061545c8f0a8b  gpt_both.ckpt
5dab10ba82cec3737c152622f477fbc9ae970f7bbd86a644b21dbce6d1392f0e  vqvae_photoshape.ckpt
ac6cdfc5f01af84a403b29f696b6a685c3752c874505e7f3d84b6efe3f17e9ef  gpt_photoshape.ckpt
```
