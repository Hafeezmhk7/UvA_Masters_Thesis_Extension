"""
Step 1 validation: read the converted transforms_train.json through the repo's
OWN readCamerasFromTransforms, render frame 0 from the pose it produces, and
save it next to the real photo. If they align, the pose-convention handling
(our y/z flip cancelling the reader's flip) is correct end to end.

Run on a GPU node from the repo root:
    python check_transforms.py --scene 0201_840151 --frame 0
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from data.vfront import readCamerasFromTransforms
from utils.render import render_core
from scenesplat_npy import load_scenesplat_scene

FT_ROOT = "/scratch-shared/mkhan4/gaussian_world/finetune_data"
GS_ROOT = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train"


def fov2focal(fov, px):
    return px / (2.0 * np.tan(fov * 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/transform_check")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    scene_dir = os.path.join(FT_ROOT, args.scene)

    # use the repo's own reader
    cams = readCamerasFromTransforms(scene_dir, "transforms_train.json",
                                     scene_dir, extension=".png")
    cam = cams[args.frame]
    # cam is a namedtuple-like with R, T, FovX, FovY, width, height, cx, cy, image_path
    R = np.array(cam.R, dtype=np.float64)   # stored transposed (glm)
    T = np.array(cam.T, dtype=np.float64)
    w, h = int(cam.width), int(cam.height)

    # rebuild world-to-camera from R,T (R is glm-transposed world2cam rotation)
    w2c = np.eye(4)
    w2c[:3, :3] = R.transpose()
    w2c[:3, 3] = T
    view = torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)

    fx = fov2focal(cam.FovX, w)
    fy = fov2focal(cam.FovY, h)
    cx = float(cam.cx)
    cy = float(cam.cy)
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]],
                     dtype=torch.float32, device=device)
    print(f"reader gave: fx={fx:.1f} fy={fy:.1f} cx={cx} cy={cy} @ {w}x{h}")

    scene = load_scenesplat_scene(os.path.join(GS_ROOT, args.scene))
    with torch.inference_mode():
        colors, _ = render_core(
            scene["coords"].to(device), scene["quats"].to(device),
            scene["scales"].to(device), scene["opacities"].to(device),
            scene["sh0"].to(device), None,
            view, K, render_size=(w, h), background_color="white",
        )
    img = (colors[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    real = np.array(Image.open(cam.image_path).convert("RGB").resize((w, h)))
    combined = np.concatenate([img, real], axis=1)
    out = os.path.join(args.out_dir, f"{args.scene}_f{args.frame}_render_vs_real.png")
    Image.fromarray(combined).save(out)
    print("saved:", out)
    print("Left = render via repo reader, right = real. They should align.")


if __name__ == "__main__":
    main()