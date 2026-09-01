"""
Render several frames of one scene at a FIXED focal length, each next to its
real photo, to confirm that a single intrinsic matches all views (not just
frame 0). If one focal matches every frame, the camera model is found. If
different frames need different focals, a single intrinsic doesn't explain the
data and the resolution/crop convention needs clarifying with the data author.

Run on a GPU node from the repo root:
    python multiframe_check.py --scene 0201_840151 --fx 270 --frames 0 50 150 300 450
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from utils.render import render_core
from scenesplat_npy import load_scenesplat_scene

CAM_ROOT = "/scratch-shared/mkhan4/gaussian_world/camera_data/scenes"
GS_ROOT = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--fx", type=float, required=True,
                   help="Focal length (px) at the resize resolution to test.")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 50, 150, 300, 450])
    p.add_argument("--out-dir", default="outputs/multiframe")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    out = os.path.join(args.out_dir, f"{args.scene}_fx{int(args.fx)}")
    os.makedirs(out, exist_ok=True)

    cam_dir = os.path.join(CAM_ROOT, args.scene)
    meta = json.load(open(os.path.join(cam_dir, "lang_feat_selected_imgs.json")))
    rw, rh = int(meta["resize"][0]), int(meta["resize"][1])
    cx, cy = rw / 2.0, rh / 2.0
    K = torch.tensor([[[args.fx, 0, cx], [0, args.fx, cy], [0, 0, 1]]],
                     dtype=torch.float32, device=device)

    scene = load_scenesplat_scene(os.path.join(GS_ROOT, args.scene))
    means = scene["coords"].to(device)
    quats = scene["quats"].to(device)
    scales = scene["scales"].to(device)
    opac = scene["opacities"].to(device)
    sh0 = scene["sh0"].to(device)
    print(f"scene {args.scene}: {means.shape[0]} gaussians, fx={args.fx}")

    frames = meta["frames"]
    for fi in args.frames:
        if fi >= len(frames):
            print(f"  frame {fi}: out of range (max {len(frames)-1})")
            continue
        fr = frames[fi]
        if fr.get("is_bad"):
            print(f"  frame {fi}: is_bad, skipping")
            continue
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32,
                            device=device).unsqueeze(0)
        with torch.inference_mode():
            colors, _ = render_core(
                means, quats, scales, opac, sh0, None,
                view, K, render_size=(rw, rh), background_color="white",
            )
        img = (colors[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        real_src = os.path.join(cam_dir, "images_unzipped", fr["file_path"])
        if os.path.exists(real_src):
            real = np.array(Image.open(real_src).convert("RGB").resize((rw, rh)))
            combined = np.concatenate([img, real], axis=1)  # render | real
            Image.fromarray(combined).save(os.path.join(out, f"f{fi:04d}_render_vs_real.png"))
        else:
            Image.fromarray(img).save(os.path.join(out, f"f{fi:04d}_render.png"))
        print(f"  frame {fi}: saved ({fr['file_path']})")

    print(f"\nSaved to {out}. Each image is [render | real] side by side.")
    print("If all frames line up at this fx, the camera model is confirmed.")


if __name__ == "__main__":
    main()