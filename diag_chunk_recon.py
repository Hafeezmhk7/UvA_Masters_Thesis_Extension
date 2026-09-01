"""
Diagnose the black reconstruction. Loads ONE chunk exactly as the training
pipeline does (VFront dataset chunk sampling), runs it through the PRETRAINED
VQ-VAE, and reports whether the decoder outputs real geometry or an empty/near-
empty tensor. Also prints the coordinate + scale ranges the encoder receives,
so we can compare against the convention that recon_scenesplat.py (which
reconstructs fine) uses.

Run on a GPU node from the repo root:
    python diag_chunk_recon.py
"""

import os
import numpy as np
import torch

from conf.dataclasses import GaussianFeatures, PCKeys
from data.common import collate_fn
from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import split_batch_dict
from scenesplat_npy import load_scenesplat_scene, chunk_scene

CKPT = "checkpoints/vqvae_both.ckpt"
GS = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train/0201_840151"


def main():
    device = torch.device("cuda")
    ae = GaussianVQVAE.load_from_checkpoint(CKPT, training_config=None).to(device).eval()

    scene = load_scenesplat_scene(GS)
    print(f"full scene: {scene['coords'].shape[0]} gaussians")

    # take one ~4.8m chunk via our own chunker (the one recon_scenesplat used)
    chunk_id, chunk = next(chunk_scene(scene, chunk_m=4.8))
    print(f"chunk {chunk_id}: {chunk['coords'].shape[0]} gaussians")
    print("  coord range:", chunk['coords'].min(0).values.tolist(),
          chunk['coords'].max(0).values.tolist())
    print("  scale (log) range:", float(chunk['scales'].min()), float(chunk['scales'].max()))
    print("  opacity (logit) range:", float(chunk['opacities'].min()), float(chunk['opacities'].max()))

    item = collate_fn([chunk])
    points, _ = split_batch_dict(item, device=device)

    with torch.inference_mode():
        tok = ae.tokenize(points, sort_latents=None)
        n_tok = tok["coords"].shape[0]
        print(f"\ntokens produced: {n_tok}")
        if n_tok == 0:
            print(">>> ENCODER produced ZERO tokens - input convention is off.")
            return
        recon = ae.decode(tok["coords"].to(device), tok["feature_ids"].to(device))
        n_out = recon[GaussianFeatures.COORDS].shape[0]
        print(f"decoded gaussians: {n_out}")
        if n_out == 0:
            print(">>> DECODER pruned everything - outputs empty (black render).")
        else:
            print(">>> Decoder produced geometry. Reconstruction is NOT black here.")
            print("  recon coord range:",
                  recon[GaussianFeatures.COORDS].min(0).values.tolist(),
                  recon[GaussianFeatures.COORDS].max(0).values.tolist())


if __name__ == "__main__":
    main()