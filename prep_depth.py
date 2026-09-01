"""
Step 2 of fine-tuning prep: render a metric depth map for every frame of a
scene by rasterizing the FULL scene's Gaussians from that frame's camera, and
write it as an .exr in the layout the repo's depth loader expects:

    <finetune_data>/<scene>/depth/<frame>.exr

The depth is gsplat's expected-depth (render_mode="RGB+ED"), which is the
camera-space z-distance per pixel, exactly what
_camera_chunk_membership_mask_from_depth unprojects with.

Reuses the transforms_train.json produced by prep_cameras.py (same poses,
same intrinsics), so depth is consistent with the RGB cameras.

Run on a GPU node from the repo root:
    python prep_depth.py --scene 0201_840151
    python prep_depth.py --all
"""

import argparse
import json
import os

import numpy as np
import torch

from utils.render import render_core
from scenesplat_npy import load_scenesplat_scene

# The transforms file has YZ_FLIP already baked in (prep_cameras applied it to
# cancel the reader's flip). The reader then does c2w[:3,1:3] *= -1 to recover
# the true pose. prep_depth does NOT go through the reader, so we must apply
# that same flip here before inverting, or the camera faces the wrong way.
READER_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

# depth-validity thresholds (metres) to reject empty-space / floater rays
MIN_DEPTH = 0.05
MAX_DEPTH = 30.0    # rooms are well under 30 m; larger = grazing ray, reject
MIN_ALPHA = 0.5     # pixel must be mostly covered by solid geometry

FT_ROOT = "/scratch-shared/mkhan4/gaussian_world/finetune_data"
GS_ROOT = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train"


def fov2focal(fov, px):
    return px / (2.0 * np.tan(fov * 0.5))


def write_depth(path, depth_hw):
    """Write a single-channel float32 depth map as .npy (no plugin deps)."""
    np.save(path, depth_hw.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch", type=int, default=8,
                    help="Frames rendered per forward pass.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")

    if args.all:
        scenes = sorted(d for d in os.listdir(FT_ROOT)
                        if os.path.isdir(os.path.join(FT_ROOT, d)))
    else:
        scenes = [args.scene] if args.scene else []
    if not scenes:
        print("No scenes. Use --scene <id> or --all.")
        return

    for scene in scenes:
        scene_dir = os.path.join(FT_ROOT, scene)
        tj = os.path.join(scene_dir, "transforms_train.json")
        if not os.path.exists(tj):
            print(f"{scene}: no transforms_train.json, run prep_cameras first. skip.")
            continue
        d = json.load(open(tj))
        W, H = int(d["w"]), int(d["h"])
        fx, fy = float(d["fl_x"]), float(d["fl_y"])
        cx, cy = float(d["cx"]), float(d["cy"])
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                         dtype=torch.float32, device=device)

        depth_dir = os.path.join(scene_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        # load full-scene gaussians once
        g = load_scenesplat_scene(os.path.join(GS_ROOT, scene))
        means = g["coords"].to(device)
        quats = g["quats"].to(device)
        scales = g["scales"].to(device)
        opac = g["opacities"].to(device)
        sh0 = g["sh0"].to(device)
        print(f"{scene}: {means.shape[0]} gaussians, {len(d['frames'])} frames @ {W}x{H}")

        frames = d["frames"]
        done = 0
        for i in range(0, len(frames), args.batch):
            batch = frames[i:i + args.batch]
            views = []
            out_paths = []
            for fr in batch:
                stem = os.path.splitext(os.path.basename(fr["file_path"]))[0]
                out_path = os.path.join(depth_dir, f"{stem}.npy")
                if os.path.exists(out_path) and not args.overwrite:
                    continue
                c2w = np.array(fr["transform_matrix"], dtype=np.float64)
                c2w = c2w.copy()
                c2w[:3, 1:3] *= -1.0   # match the reader's flip (recovers true pose)
                views.append(np.linalg.inv(c2w))
                out_paths.append(out_path)
            if not views:
                continue
            view = torch.tensor(np.stack(views), dtype=torch.float32, device=device)
            Ks = K.unsqueeze(0).repeat(view.shape[0], 1, 1)
            with torch.inference_mode():
                colors, aux = render_core(
                    means, quats, scales, opac, sh0, None,
                    view, Ks, render_size=(W, H),
                    background_color="white", render_mode="RGB+ED",
                )
            # colors: (C, 4, H, W); channel 3 is accumulated (alpha-weighted) depth.
            if colors.shape[1] < 4:
                raise RuntimeError(
                    "render did not return a depth channel; gsplat backend may "
                    "not support RGB+ED. Channels=" + str(colors.shape[1]))
            acc_depth = colors[:, 3:4]                      # (C,1,H,W) accumulated depth
            alphas = aux["alphas"]                          # (C,1,H,W) accumulated alpha
            # normalize to true mean surface depth in metres; 0 where nothing hit
            true_depth = torch.where(alphas > 1e-6, acc_depth / alphas.clamp_min(1e-6),
                                     torch.zeros_like(acc_depth))
            # Clamp out non-physical depths (rays grazing empty space / faint
            # floaters accumulate huge or ~0 depth). Anything outside a sane
            # metric range is set to 0, which the mask function treats as
            # "no valid surface" and excludes. Also require enough opacity that
            # the pixel actually hit solid geometry.
            valid = (alphas > MIN_ALPHA) & (true_depth > MIN_DEPTH) & (true_depth < MAX_DEPTH)
            true_depth = torch.where(valid, true_depth, torch.zeros_like(true_depth))
            depth = true_depth[:, 0].cpu().numpy()          # (C, H, W)
            for k, out_path in enumerate(out_paths):
                write_depth(out_path, depth[k])
                done += 1
            print(f"  {scene}: {done} depth maps written", end="\r")
        print(f"\n{scene}: done, {done} new depth maps in {depth_dir}")


if __name__ == "__main__":
    main()