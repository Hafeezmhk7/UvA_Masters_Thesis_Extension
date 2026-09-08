"""
Color-statistics diagnostic: distinguishes two failure modes that look
identical in a viewer but call for opposite fixes.

  1. Variance collapse (mean-regression): recon color mean tracks GT closely,
     but recon variance/saturation is well below GT. This is the discrete
     12-bit-bottleneck hedging toward "safe average" appearance predicted by
     the L1/VGG-regression-loss argument -> the fix is a generative decoder
     (WeTok-style noise-conditioned decoding), not a bug fix.
  2. Systematic mean shift: recon color is offset from GT even in the mean.
     This points at a concrete bug candidate: the sh2rgb/rgb2sh round-trip in
     ClampedRGBColorRepresentation, or the sh0 head's 0.5-grey const_init_bias
     (conf/internal_representations.py) never being corrected away from grey.
     No amount of adversarial/generative decoder work fixes this; it needs a
     direct fix.

Reuses the same scene loading / chunked reconstruction as eval_quant.py so
this is apples-to-apples with your PSNR/LPIPS numbers. Masks to pixels where
BOTH the GT and reconstruction actually render a Gaussian (alpha > thresh),
so background (white) pixels don't wash out the statistics.

Run on a GPU node from the repo root:
    python color_stats.py \
        --scenes 0390_841511 0391_841500 0392_841498 \
        --checkpoint logs/scenesplat_ft200/lightning_logs/version_X/checkpoints/<ckpt> \
        --cams-per-scene 8
"""

import argparse
import json
import os

import numpy as np
import torch

from eval_quant import FT_ROOT, GS_ROOT, reconstruct_full
from model.gaussian_vqvae import GaussianVQVAE
from scenesplat_npy import load_scenesplat_scene
from utils.render import render_core


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


def hsv_saturation(rgb):
    # rgb: (3,H,W) in [0,1] -> saturation (H,W), standard max/min definition
    maxc = rgb.max(0).values
    minc = rgb.min(0).values
    return torch.where(maxc > 1e-6, (maxc - minc) / maxc.clamp(min=1e-6), torch.zeros_like(maxc))


def stats_for(rgb, hit_mask):
    px = rgb[:, hit_mask]  # (3, N)
    mean = px.mean(dim=1).cpu().numpy()
    std = px.std(dim=1).cpu().numpy()
    sat = float(hsv_saturation(rgb)[hit_mask].mean())
    return mean, std, sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cams-per-scene", type=int, default=8)
    ap.add_argument("--chunk-m", type=float, default=4.0)
    ap.add_argument("--alpha-thresh", type=float, default=0.5)
    ap.add_argument("--label", default=None,
                     help="Tag for this run (e.g. 'baseline_1tok', 'rvq4_epoch9') stored in --save-json output.")
    ap.add_argument("--save-json", default=None,
                     help="Write the aggregate numbers to this path for use with compare_color_stats.py.")
    args = ap.parse_args()

    device = torch.device("cuda")
    ae = GaussianVQVAE.load_from_checkpoint(args.checkpoint, training_config=None).to(device).eval()

    gt_means, gt_stds, gt_sats = [], [], []
    re_means, re_stds, re_sats = [], [], []

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
            if int(hit.sum()) < 100:
                continue

            m, s, sat = stats_for(gt_rgb, hit)
            gt_means.append(m); gt_stds.append(s); gt_sats.append(sat)
            m, s, sat = stats_for(re_rgb, hit)
            re_means.append(m); re_stds.append(s); re_sats.append(sat)
        print(f"  {scene_id} done ({len(idxs)} cams)")

    if not gt_means:
        print("No valid camera/scene pairs produced overlapping hit pixels; nothing to report.")
        return

    gt_mean, re_mean = np.mean(gt_means, axis=0), np.mean(re_means, axis=0)
    gt_std, re_std = np.mean(gt_stds, axis=0), np.mean(re_stds, axis=0)
    gt_sat, re_sat = float(np.mean(gt_sats)), float(np.mean(re_sats))

    print("\n" + "=" * 60)
    print("COLOR STATISTICS (over pixels both GT and recon actually render)")
    print("=" * 60)
    print(f"{'':10s} {'R':>8s} {'G':>8s} {'B':>8s}   sat")
    print(f"{'GT mean':10s} {gt_mean[0]:8.4f} {gt_mean[1]:8.4f} {gt_mean[2]:8.4f}   {gt_sat:.4f}")
    print(f"{'recon mean':10s} {re_mean[0]:8.4f} {re_mean[1]:8.4f} {re_mean[2]:8.4f}   {re_sat:.4f}")
    print(f"{'GT std':10s} {gt_std[0]:8.4f} {gt_std[1]:8.4f} {gt_std[2]:8.4f}")
    print(f"{'recon std':10s} {re_std[0]:8.4f} {re_std[1]:8.4f} {re_std[2]:8.4f}")

    mean_shift = np.abs(re_mean - gt_mean)
    std_ratio = re_std / np.clip(gt_std, 1e-6, None)
    sat_ratio = re_sat / max(gt_sat, 1e-6)
    print(f"\nmean shift per channel:      {mean_shift.round(4).tolist()}")
    print(f"std ratio (recon/GT):        {std_ratio.round(3).tolist()}")
    print(f"saturation ratio (recon/GT): {sat_ratio:.3f}")

    # Report the gap to parity as a continuous number rather than a pass/fail
    # verdict against a fixed threshold: a run at 0.849 and a run at 0.851
    # are the same regime, and a hard cutoff (e.g. 0.85) makes them look
    # qualitatively different when they aren't. Use compare_color_stats.py
    # (with --save-json below) to see whether a *change* between two runs
    # is real, rather than reading a single run's ratio against a cutoff.
    std_gap = 1.0 - float(std_ratio.mean())
    sat_gap = 1.0 - sat_ratio
    print("\nGap to parity (0 = matches GT, no threshold implied):")
    print(f"  color-variance gap:  {std_gap:.3f}  (mean std ratio {std_ratio.mean():.3f})")
    print(f"  saturation gap:      {sat_gap:.3f}  (saturation ratio {sat_ratio:.3f})")
    print(f"  max |mean shift|:    {float(mean_shift.max()):.4f}")
    if mean_shift.mean() >= 0.05:
        print(
            "\nNote: mean shift is large enough to check direction before assuming "
            "capacity is the issue -- if reconstructions shift *toward* 0.5 grey "
            "across channels, check ClampedRGBColorRepresentation's sh2rgb/rgb2sh "
            "round-trip and the sh0 head's 0.5-grey const_init_bias "
            "(conf/internal_representations.py); a shift *away* from grey rules that "
            "bug out, as it did on this codebase's first run."
        )

    if args.save_json:
        payload = {
            "label": args.label,
            "checkpoint": args.checkpoint,
            "scenes": args.scenes,
            "n_pixels_pairs": len(gt_means),
            "gt_mean": gt_mean.tolist(),
            "recon_mean": re_mean.tolist(),
            "gt_std": gt_std.tolist(),
            "recon_std": re_std.tolist(),
            "gt_saturation": gt_sat,
            "recon_saturation": re_sat,
            "mean_shift_per_channel": mean_shift.tolist(),
            "std_ratio_per_channel": std_ratio.tolist(),
            "std_ratio_mean": float(std_ratio.mean()),
            "saturation_ratio": sat_ratio,
        }
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {args.save_json}")


if __name__ == "__main__":
    main()
