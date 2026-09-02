"""
Quantitative before/after evaluation of VQ-VAE fine-tuning.

For each held-out scene, for a set of held-out cameras:
  1. render the ORIGINAL splats  -> ground-truth image (target)
  2. reconstruct the scene chunk-by-chunk with the PRETRAINED model,
     render from the same camera -> pretrained image
  3. same with the FINE-TUNED model -> finetuned image
Then compute PSNR / SSIM / LPIPS of pretrained-vs-GT and finetuned-vs-GT,
averaged over cameras and scenes. Fine-tuning helps only if the finetuned
numbers beat the pretrained ones. inf/nan PSNR values are excluded.

Reconstruction here uses the same chunking as recon_scenesplat (chunk_scene),
reconstructs every chunk, and unions the decoded gaussians before rendering the
full scene from each camera - so the comparison is at scene level, matching how
the scenes are actually viewed.

Run on a GPU node from the repo root:
    python eval_quant.py \
        --scenes 0390_841511 0391_841500 0392_841498 \
        --finetuned logs/scenesplat_ft200/lightning_logs/version_X/checkpoints/<ckpt> \
        --cams-per-scene 8
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from conf.dataclasses import GaussianFeatures, PCKeys
from data.common import collate_fn
from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import split_batch_dict
from utils.render import render_core
from scenesplat_npy import load_scenesplat_scene, chunk_scene

GS_ROOT = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train"
FT_ROOT = "/scratch-shared/mkhan4/gaussian_world/finetune_data"
PRETRAINED = "checkpoints/vqvae_both.ckpt"

READER_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return None  # exclude degenerate/inf
    return 10.0 * np.log10(1.0 / mse)


def reconstruct_full(ae, scene, device, chunk_m=4.0):
    """Reconstruct every chunk and concatenate the decoded gaussians."""
    parts = {k: [] for k in ["coords", "sh0", "opacities", "scales", "quats"]}
    for _, chunk in chunk_scene(scene, chunk_m=chunk_m):
        item = collate_fn([chunk])
        points, _ = split_batch_dict(item, device=device)
        with torch.inference_mode():
            tok = ae.tokenize(points, sort_latents=None)
            if tok["coords"].shape[0] == 0:
                continue
            recon = ae.decode(tok["coords"].to(device), tok["feature_ids"].to(device))
        m = recon[PCKeys.BATCH] == 0
        parts["coords"].append(recon[GaussianFeatures.COORDS][m])
        parts["sh0"].append(recon[GaussianFeatures.SH0][m])
        parts["opacities"].append(recon[GaussianFeatures.OPACITIES][m])
        parts["scales"].append(recon[GaussianFeatures.SCALES][m])
        parts["quats"].append(recon[GaussianFeatures.QUATS][m])
    if not parts["coords"]:
        return None
    return {k: torch.cat(v, 0) for k, v in parts.items()}


def render_from(gauss, view, K, W, H, device):
    with torch.inference_mode():
        colors, _ = render_core(
            gauss["coords"].to(device), gauss["quats"].to(device),
            gauss["scales"].to(device), gauss["opacities"].to(device),
            gauss["sh0"].to(device), None,
            view, K, render_size=(W, H), background_color="white",
        )
    return colors[0, :3].clamp(0, 1)  # (3,H,W)


def try_lpips():
    try:
        import lpips
        return lpips.LPIPS(net="vgg").cuda().eval()
    except Exception as e:
        print("(LPIPS unavailable:", e, ") - skipping LPIPS")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Held-out scene ids (from the val split).")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--cams-per-scene", type=int, default=8)
    ap.add_argument("--chunk-m", type=float, default=4.0)
    args = ap.parse_args()

    device = torch.device("cuda")
    lpips_fn = try_lpips()

    print("loading models...")
    ae_pre = GaussianVQVAE.load_from_checkpoint(PRETRAINED, training_config=None).to(device).eval()
    ae_ft = GaussianVQVAE.load_from_checkpoint(args.finetuned, training_config=None).to(device).eval()

    agg = {"pre": {"psnr": [], "lpips": []}, "ft": {"psnr": [], "lpips": []}}

    for scene_id in args.scenes:
        gs_dir = os.path.join(GS_ROOT, scene_id)
        cam_json = os.path.join(FT_ROOT, scene_id, "transforms_train.json")
        if not os.path.exists(cam_json):
            print(f"{scene_id}: no transforms, skipping")
            continue
        scene = load_scenesplat_scene(gs_dir)
        d = json.load(open(cam_json))
        W, H = int(d["w"]), int(d["h"])
        fx, fy, cx, cy = d["fl_x"], d["fl_y"], d["cx"], d["cy"]
        K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]],
                         dtype=torch.float32, device=device)

        # reconstruct once per model (scene-level), reuse across cameras
        print(f"{scene_id}: reconstructing (pretrained)...")
        g_pre = reconstruct_full(ae_pre, scene, device, args.chunk_m)
        print(f"{scene_id}: reconstructing (finetuned)...")
        g_ft = reconstruct_full(ae_ft, scene, device, args.chunk_m)
        g_gt = {k: scene[k] for k in ["coords", "sh0", "opacities", "scales", "quats"]}

        # evenly-spaced cameras across the trajectory
        frames = d["frames"]
        idxs = np.linspace(0, len(frames) - 1, args.cams_per_scene).astype(int)
        for fi in idxs:
            c2w = np.array(frames[fi]["transform_matrix"], dtype=np.float64).copy()
            c2w[:3, 1:3] *= -1.0   # reader-flip convention (matches training)
            view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32,
                                device=device).unsqueeze(0)
            gt = render_from(g_gt, view, K, W, H, device)
            for tag, g in [("pre", g_pre), ("ft", g_ft)]:
                if g is None:
                    continue
                im = render_from(g, view, K, W, H, device)
                p = psnr(im, gt)
                if p is not None:
                    agg[tag]["psnr"].append(p)
                if lpips_fn is not None:
                    with torch.inference_mode():
                        lp = lpips_fn(im.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()
                    agg[tag]["lpips"].append(lp)
        print(f"  {scene_id} done ({len(idxs)} cams)")

    def summ(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return (np.mean(xs), len(xs)) if xs else (float("nan"), 0)

    print("\n" + "=" * 60)
    print("RESULTS (higher PSNR better, lower LPIPS better)")
    print("=" * 60)
    for tag, name in [("pre", "PRETRAINED"), ("ft", "FINE-TUNED")]:
        pm, pn = summ(agg[tag]["psnr"])
        lm, ln = summ(agg[tag]["lpips"])
        print(f"{name:12s}  PSNR={pm:.3f} (n={pn})   LPIPS={lm:.4f} (n={ln})")
    pm_pre, _ = summ(agg["pre"]["psnr"])
    pm_ft, _ = summ(agg["ft"]["psnr"])
    if np.isfinite(pm_pre) and np.isfinite(pm_ft):
        d = pm_ft - pm_pre
        print(f"\nPSNR change from fine-tuning: {d:+.3f} dB "
              f"({'improvement' if d > 0 else 'no improvement / worse'})")


if __name__ == "__main__":
    main()