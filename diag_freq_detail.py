"""
Spatial-frequency / blur diagnostic -- the direct follow-up to
diag_color_stats.py's own conclusion: global colour statistics (mean, std,
saturation) were close-ish to GT (mild collapse, not dominant), so the
perceived "washed out" look is more likely spatially-structured detail loss
(blur) than a colour bug. This script measures that directly instead of by
eye.

For each rendered GT/recon pair (same cameras, same alpha-intersection
masking as diag_color_stats.py), computes:
  - Laplacian variance ratio (recon/gt): classic blur metric, lower = blurrier.
  - mean gradient-magnitude ratio (recon/gt): a second, independent blur check.
  - radially-averaged 2D power spectrum, binned into low/mid/high frequency
    thirds (each normalized to sum to 1, so this compares the SHAPE of the
    spectrum, not overall contrast).

Why the frequency bands matter more than a single blur number: if the
bottleneck is genuinely bits-per-spatial-region (the 12-bit-per-20cm-voxel
argument), the recon/gt energy ratio should stay near 1 at low frequencies
and drop sharply at high frequencies -- a cutoff, not a uniform rolloff.
A uniform rolloff across all bands would instead suggest a generic smoothing
effect (e.g. from the render/mask pipeline) rather than a genuine information
ceiling in the tokenizer.

Run on a GPU node from the repo root:
    python diag_freq_detail.py \
        --scenes $(head -5 data_splits/scenesplat700_val.txt) \
        --checkpoint logs/scenesplat_rvq4_long/lightning_logs/version_X/checkpoints/<ckpt> \
        --cams-per-scene 6
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval_quant import FT_ROOT, GS_ROOT, reconstruct_full
from model.gaussian_vqvae import GaussianVQVAE
from scenesplat_npy import load_scenesplat_scene
from utils.render import render_core

LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def render_with_alpha(gauss, view, K, W, H, device):
    with torch.inference_mode():
        colors, aux = render_core(
            gauss["coords"].to(device), gauss["quats"].to(device),
            gauss["scales"].to(device), gauss["opacities"].to(device),
            gauss["sh0"].to(device), None,
            view, K, render_size=(W, H), background_color="white",
            render_mode="RGB+ED",
        )
    rgb = colors[0, :3].clamp(0, 1)  # (3,H,W)
    alpha = aux["alphas"][0, 0]  # (H,W)
    return rgb, alpha


def fill_background(rgb, hit_mask):
    """Replace non-hit pixels with the hit-region mean so the fill is smooth
    and doesn't inject spurious edges at the mask boundary that would
    contaminate the blur/frequency metrics.
    """
    out = rgb.clone()
    if hit_mask.any():
        mean = rgb[:, hit_mask].mean(dim=1, keepdim=True)
    else:
        mean = rgb.mean(dim=(1, 2), keepdim=True)
    out[:, ~hit_mask] = mean.squeeze(-1)
    return out


def to_gray(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def laplacian_variance(gray):
    k = LAPLACIAN_KERNEL.to(gray.device, gray.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(gray[None, None], k, padding=1)
    return float(lap.var())


def mean_gradient_magnitude(gray):
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    h, w = gray.shape
    return float(torch.sqrt(gx[: h - 1, :] ** 2 + gy[:, : w - 1] ** 2).mean())


def radial_band_energy(gray, n_bands=3):
    H, W = gray.shape
    spec = torch.fft.fftshift(torch.fft.fft2(gray))
    power = spec.abs() ** 2
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=gray.device),
        torch.linspace(-1, 1, W, device=gray.device),
        indexing="ij",
    )
    r = torch.sqrt(xx**2 + yy**2)
    r = r / r.max()
    edges = torch.linspace(0, 1, n_bands + 1)
    total = power.sum().clamp(min=1e-12)
    bands = []
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        m = (r >= lo) & (r <= hi if i == n_bands - 1 else r < hi)
        bands.append(float(power[m].sum() / total))
    return bands  # [low, mid, high] fractions, sum to 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cams-per-scene", type=int, default=8)
    ap.add_argument("--chunk-m", type=float, default=4.0)
    ap.add_argument("--alpha-thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda")
    ae = GaussianVQVAE.load_from_checkpoint(args.checkpoint, training_config=None).to(device).eval()

    lap_gt, lap_re = [], []
    grad_gt, grad_re = [], []
    bands_gt, bands_re = [], []

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
        K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device=device)

        print(f"{scene_id}: reconstructing...")
        g_re = reconstruct_full(ae, scene, device, args.chunk_m)
        if g_re is None:
            print(f"{scene_id}: empty reconstruction, skipping")
            continue
        g_gt = {k: scene[k] for k in ["coords", "sh0", "opacities", "scales", "quats"]}

        frames = d["frames"]
        idxs = np.linspace(0, len(frames) - 1, args.cams_per_scene).astype(int)
        for fi in idxs:
            c2w = np.array(frames[fi]["transform_matrix"], dtype=np.float64).copy()
            c2w[:3, 1:3] *= -1.0  # reader-flip convention (matches training)
            view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device).unsqueeze(0)

            gt_rgb, gt_alpha = render_with_alpha(g_gt, view, K, W, H, device)
            re_rgb, re_alpha = render_with_alpha(g_re, view, K, W, H, device)
            hit = (gt_alpha > args.alpha_thresh) & (re_alpha > args.alpha_thresh)
            if int(hit.sum()) < 1000:
                continue

            gt_gray = to_gray(fill_background(gt_rgb, hit))
            re_gray = to_gray(fill_background(re_rgb, hit))

            lap_gt.append(laplacian_variance(gt_gray))
            lap_re.append(laplacian_variance(re_gray))
            grad_gt.append(mean_gradient_magnitude(gt_gray))
            grad_re.append(mean_gradient_magnitude(re_gray))
            bands_gt.append(radial_band_energy(gt_gray))
            bands_re.append(radial_band_energy(re_gray))
        print(f"  {scene_id} done ({len(idxs)} cams)")

    if not lap_gt:
        print("No valid camera/scene pairs; nothing to report.")
        return

    lap_ratio = float(np.mean(lap_re) / max(np.mean(lap_gt), 1e-12))
    grad_ratio = float(np.mean(grad_re) / max(np.mean(grad_gt), 1e-12))
    bands_gt_m = np.mean(bands_gt, axis=0)
    bands_re_m = np.mean(bands_re, axis=0)
    band_ratio = bands_re_m / np.clip(bands_gt_m, 1e-12, None)

    print("\n" + "=" * 60)
    print("SPATIAL-FREQUENCY / BLUR DIAGNOSTIC")
    print("=" * 60)
    print(f"Laplacian variance:  gt={np.mean(lap_gt):.5f}  recon={np.mean(lap_re):.5f}  ratio={lap_ratio:.3f}")
    print(f"Mean grad magnitude: gt={np.mean(grad_gt):.5f}  recon={np.mean(grad_re):.5f}  ratio={grad_ratio:.3f}")
    print("\nPower-spectrum energy fraction by radial band (low/mid/high, each row sums to 1):")
    print(f"  gt:    {bands_gt_m.round(4).tolist()}")
    print(f"  recon: {bands_re_m.round(4).tolist()}")
    print(f"  ratio (recon/gt) per band: {band_ratio.round(3).tolist()}")

    print("\nVerdict:")
    high_band_ratio = band_ratio[-1]
    if lap_ratio < 0.7 or grad_ratio < 0.7 or high_band_ratio < 0.7:
        print(
            "  -> substantial blur: high-frequency/edge content is well below GT. "
            "Combined with the earlier mild color-collapse result, this points at "
            "spatial/quantization capacity (the bits-per-20cm-region bottleneck) as "
            "the dominant failure mode, not color per se -- consistent with the RVQ4 "
            "gain coming from recovered detail, not recovered color."
        )
        if high_band_ratio < band_ratio[0] - 0.1:
            print(
                "     The high band drops noticeably more than the low band -- a "
                "cutoff shape, not a uniform rolloff -- which is the specific "
                "signature of a bit-budget ceiling rather than a generic blur filter."
            )
    else:
        print(
            "  -> blur is present but moderate at these aggregate metrics; inspect "
            "the per-band spectrum ratio directly -- a sharp high-band dropoff even "
            "with a mild overall ratio is still the bottleneck signature."
        )


if __name__ == "__main__":
    main()
