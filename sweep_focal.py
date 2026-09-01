"""
Sweep focal length to find the intrinsics that reproduce the real photo's FOV.

The stored intrinsics are internally inconsistent (fl_x for one resolution,
cx/cy for another), and the first render came out too zoomed-in, meaning the
focal was too long. This renders frame 0 at several focal lengths so you can
eyeball which one matches the real photo's width.

Run on a GPU node from the repo root:
    python sweep_focal.py --scene 0201_840151 --frame 0
Then compare the outputs in outputs/focal_sweep/ against the real photo.
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
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/focal_sweep")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    cam_dir = os.path.join(CAM_ROOT, args.scene)
    meta = json.load(open(os.path.join(cam_dir, "lang_feat_selected_imgs.json")))

    rw, rh = int(meta["resize"][0]), int(meta["resize"][1])
    cx, cy = rw / 2.0, rh / 2.0

    frame = meta["frames"][args.frame]
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    view = torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)

    scene = load_scenesplat_scene(os.path.join(GS_ROOT, args.scene))
    means = scene["coords"].to(device)
    quats = scene["quats"].to(device)
    scales = scene["scales"].to(device)
    opac = scene["opacities"].to(device)
    sh0 = scene["sh0"].to(device)

    # candidate focal lengths (px) at the resize resolution
    candidates = [300.0, 290.0, 280.0, 270.0, 277.0]

    for fx in candidates:
        K = torch.tensor([[[fx, 0, cx], [0, fx, cy], [0, 0, 1]]],
                         dtype=torch.float32, device=device)
        with torch.inference_mode():
            colors, _ = render_core(
                means, quats, scales, opac, sh0, None,
                view, K, render_size=(rw, rh),
                background_color="white",
            )
        img = (colors[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        path = os.path.join(args.out_dir, f"{args.scene}_f{args.frame}_fx{int(fx)}.png")
        Image.fromarray(img).save(path)
        fov = 2 * np.degrees(np.arctan(rw / (2 * fx)))
        print(f"fx={fx:.0f}  FOV={fov:.1f}deg  ->  {path}")

    # also save the real photo at matching resolution
    real_src = os.path.join(cam_dir, "images_unzipped", frame["file_path"])
    if os.path.exists(real_src):
        Image.open(real_src).convert("RGB").resize((rw, rh)).save(
            os.path.join(args.out_dir, f"{args.scene}_f{args.frame}_REAL.png"))
        print("real photo saved for comparison")

    print("\nCompare each fx render to _REAL.png. Pick the fx whose width/FOV matches.")


if __name__ == "__main__":
    main()