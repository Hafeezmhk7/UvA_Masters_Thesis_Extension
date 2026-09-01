"""
Validate the SceneSplat camera poses against the Gaussian splats.

Renders the full-scene splats from ONE real camera pose (frame 0), using
intrinsics normalized to the 960x540 resize, and saves the render next to the
downsized real photo. If they line up (same walls, furniture, orientation),
the intrinsics and pose convention are correct and fine-tuning can proceed.
If the render is flipped/rotated, we apply the OpenCV<->OpenGL axis fix.

Run from the GaussianGPT repo root:
    python validate_camera.py --scene 0201_840151 --frame 0
Optionally test the axis flip:
    python validate_camera.py --scene 0201_840151 --frame 0 --flip-yz
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

# OpenCV (+x right, +y down, +z forward) vs OpenGL/gsplat convention differ by
# flipping y and z. This diag matrix applies that flip to a c2w matrix.
YZ_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--flip-yz", action="store_true",
                   help="Apply OpenCV<->OpenGL y/z flip to the pose.")
    p.add_argument("--out-dir", default="outputs/cam_check")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    cam_dir = os.path.join(CAM_ROOT, args.scene)
    meta = json.load(open(os.path.join(cam_dir, "lang_feat_selected_imgs.json")))

    # --- intrinsics, normalized to the resize resolution (960x540) ---
    rw, rh = int(meta["resize"][0]), int(meta["resize"][1])
    fx = float(meta["fl_x"])   # already for the resize resolution
    fy = float(meta["fl_y"])
    cx = rw / 2.0              # force centered principal point at resize res
    cy = rh / 2.0
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]],
                     dtype=torch.float32, device=device)
    print(f"intrinsics @ {rw}x{rh}: fx={fx:.1f} fy={fy:.1f} cx={cx} cy={cy}")

    # --- the chosen frame's pose ---
    frame = meta["frames"][args.frame]
    print(f"frame {args.frame}: {frame['file_path']}  is_bad={frame.get('is_bad')}")
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    if args.flip_yz:
        c2w = c2w @ YZ_FLIP
        print("applied y/z flip")
    w2c = np.linalg.inv(c2w)
    view = torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)

    # --- load full-scene splats (pre-activation, matches render_core) ---
    scene = load_scenesplat_scene(os.path.join(GS_ROOT, args.scene))
    means = scene["coords"].to(device)
    quats = scene["quats"].to(device)
    scales = scene["scales"].to(device)
    opac = scene["opacities"].to(device)
    sh0 = scene["sh0"].to(device)
    print(f"scene has {means.shape[0]} gaussians")

    with torch.inference_mode():
        colors, _ = render_core(
            means, quats, scales, opac, sh0, None,
            view, K, render_size=(rw, rh),
            background_color="white",
        )
    img = (colors[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    tag = "flip" if args.flip_yz else "noflip"
    render_path = os.path.join(args.out_dir, f"{args.scene}_f{args.frame}_render_{tag}.png")
    Image.fromarray(img).save(render_path)
    print("saved render:", render_path)

    # --- downsize the real photo to the same resolution for comparison ---
    real_zip_dir = os.path.join(cam_dir, "images_unzipped", frame["file_path"])
    if os.path.exists(real_zip_dir):
        real = Image.open(real_zip_dir).convert("RGB").resize((rw, rh))
        real_path = os.path.join(args.out_dir, f"{args.scene}_f{args.frame}_real.png")
        real.save(real_path)
        print("saved real  :", real_path)
    else:
        print(f"(real image not found at {real_zip_dir}; unzip images.zip first)")

    print("\nCompare the two PNGs. If they show the same view, poses are correct.")
    print("If the render is flipped/rotated, re-run with --flip-yz and compare.")


if __name__ == "__main__":
    main()