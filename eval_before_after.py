"""
Before/after evaluation: reconstruct a HELD-OUT scene with both the pretrained
checkpoint and a fine-tuned checkpoint, saving each chunk's reconstruction as a
.ply so they can be compared side by side in a 3DGS viewer.

Use a scene NOT in the training split (e.g. 0207_840167 or 0208_840166).

Run on a GPU node from the repo root:
    python eval_before_after.py \
        --scene 0207_840167 \
        --finetuned logs/scenesplat_ft200/lightning_logs/version_26301338/checkpoints/loss_monitor_epoch_14_step_2700.ckpt
"""

import argparse
import os

import torch

from conf.dataclasses import GaussianFeatures, PCKeys
from data.common import collate_fn
from data.photoshape import save_inria_ply
from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import split_batch_dict
from scenesplat_npy import load_scenesplat_scene, chunk_scene

GS_ROOT = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train"
PRETRAINED = "checkpoints/vqvae_both.ckpt"


def payload(points, batch_idx=0):
    m = points[PCKeys.BATCH] == batch_idx
    return {
        "coords": points[GaussianFeatures.COORDS][m].cpu(),
        "sh0": points[GaussianFeatures.SH0][m].cpu(),
        "opacities": points[GaussianFeatures.OPACITIES][m].cpu(),
        "scales": points[GaussianFeatures.SCALES][m].cpu(),
        "quats": points[GaussianFeatures.QUATS][m].cpu(),
    }


def reconstruct(ae, chunk, device):
    item = collate_fn([chunk])
    points, _ = split_batch_dict(item, device=device)
    with torch.inference_mode():
        tok = ae.tokenize(points, sort_latents=None)
        if tok["coords"].shape[0] == 0:
            return None, 0
        recon = ae.decode(tok["coords"].to(device), tok["feature_ids"].to(device))
    p = payload(recon)
    return p, p["coords"].shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--finetuned", required=True,
                    help="Path to a fine-tuned .ckpt")
    ap.add_argument("--chunk-m", type=float, default=4.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    out_dir = args.out_dir or f"outputs/before_after/{args.scene}"
    os.makedirs(out_dir, exist_ok=True)

    scene = load_scenesplat_scene(os.path.join(GS_ROOT, args.scene))
    print(f"scene {args.scene}: {scene['coords'].shape[0]} gaussians")

    print("loading pretrained...")
    ae_pre = GaussianVQVAE.load_from_checkpoint(
        PRETRAINED, training_config=None).to(device).eval()
    print("loading fine-tuned...")
    ae_ft = GaussianVQVAE.load_from_checkpoint(
        args.finetuned, training_config=None).to(device).eval()

    for chunk_id, chunk in chunk_scene(scene, chunk_m=args.chunk_m):
        # ground truth for this chunk
        save_inria_ply(
            output_path=os.path.join(out_dir, f"{chunk_id}_gt.ply"),
            coords=chunk["coords"], sh0=chunk["sh0"],
            opacities=chunk["opacities"], scales=chunk["scales"],
            quats=chunk["quats"],
        )
        p_pre, n_pre = reconstruct(ae_pre, chunk, device)
        p_ft, n_ft = reconstruct(ae_ft, chunk, device)
        if p_pre is not None:
            save_inria_ply(output_path=os.path.join(out_dir, f"{chunk_id}_pretrained.ply"),
                           **p_pre)
        if p_ft is not None:
            save_inria_ply(output_path=os.path.join(out_dir, f"{chunk_id}_finetuned.ply"),
                           **p_ft)
        print(f"  {chunk_id}: in={chunk['coords'].shape[0]} "
              f"pretrained_out={n_pre} finetuned_out={n_ft}")

    print(f"\nSaved to {out_dir}.")
    print("For each chunk compare _gt.ply / _pretrained.ply / _finetuned.ply in a viewer.")
    print("Fine-tuning helps if _finetuned is closer to _gt (color, detail) than _pretrained.")


if __name__ == "__main__":
    main()