"""
Step 2 validation: prove the generated depth maps produce correct chunk masks.

Picks a cube somewhere in the scene, then for a chosen frame:
  - loads that frame's generated depth .exr
  - calls the repo's OWN _camera_chunk_membership_mask_from_depth
  - overlays the resulting mask (red) on the real photo

If the red region lands on the part of the room inside the cube, depth +
poses + intrinsics are all consistent and chunk masking works. This is the
gate before trusting depth-masked fine-tuning.

Run on a GPU node from the repo root:
    python check_depth_mask.py --scene 0201_840151 --frame 0
Tune the cube with --center and --half if the default lands on empty space.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from data.vfront import readCamerasFromTransforms
from data.vfront_dataset import _camera_chunk_membership_mask_from_depth

FT_ROOT = "/scratch-shared/mkhan4/gaussian_world/finetune_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--center", type=float, nargs=3, default=None,
                    help="Cube center xyz in world coords. Default = scene bbox center.")
    ap.add_argument("--half", type=float, default=2.0,
                    help="Half-extent of the cube in meters (default 2.0 -> 4m cube).")
    ap.add_argument("--out-dir", default="outputs/depth_mask_check")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    scene_dir = os.path.join(FT_ROOT, args.scene)

    d = json.load(open(os.path.join(scene_dir, "transforms_train.json")))
    W, H = int(d["w"]), int(d["h"])
    cx, cy = float(d["cx"]), float(d["cy"])

    cams = readCamerasFromTransforms(scene_dir, "transforms_train.json",
                                     scene_dir, extension=".png")
    cam = cams[args.frame]
    stem = os.path.splitext(os.path.basename(cam.image_path))[0]

    depth_path = os.path.join(scene_dir, "depth", f"{stem}.npy")
    depth_np = np.load(depth_path)
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
    depth = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)  # (1,H,W)
    print(f"depth {depth.shape}, range [{depth_np.min():.2f}, {depth_np.max():.2f}] m")

    # choose a cube
    if args.center is None:
        # camera position (c2w translation) pushed forward a bit is a decent guess;
        # here we just use the depth median along the center ray to place the cube.
        center = None
    if args.center is not None:
        cxyz = np.array(args.center, dtype=np.float64)
    else:
        # place cube at the point the center pixel sees
        zc = float(depth_np[H // 2, W // 2])
        fx = 0.5 * W / np.tan(0.5 * cam.FovX)
        fy = 0.5 * H / np.tan(0.5 * cam.FovY)
        x_cam = (W / 2 - cx) * zc / fx
        y_cam = (H / 2 - cy) * zc / fy
        p_cam = np.array([x_cam, y_cam, zc, 1.0])
        # cam.R is glm-transposed w2c rotation; rebuild c2w
        w2c = np.eye(4); w2c[:3, :3] = np.array(cam.R).T; w2c[:3, 3] = cam.T
        c2w = np.linalg.inv(w2c)
        cxyz = (c2w @ p_cam)[:3]
    half = args.half
    chunk_bounds = ([float(cxyz[0] - half), float(cxyz[1] - half), float(cxyz[2] - half)],
                    [float(cxyz[0] + half), float(cxyz[1] + half), float(cxyz[2] + half)])
    print(f"cube center {cxyz.round(2)}, half {half} -> bounds {chunk_bounds}")

    mask = _camera_chunk_membership_mask_from_depth(
        cam, depth, chunk_bounds,
        width=W, height=H, cx=cx, cy=cy, eps=1e-4,
    )  # (1,H,W) bool
    m = mask[0].cpu().numpy().astype(bool)
    print(f"mask covers {m.mean()*100:.1f}% of pixels")

    real = np.array(Image.open(cam.image_path).convert("RGB").resize((W, H))).astype(np.float32)
    overlay = real.copy()
    overlay[m] = 0.5 * overlay[m] + 0.5 * np.array([255, 0, 0])  # red where masked
    combined = np.concatenate([real.astype(np.uint8), overlay.astype(np.uint8)], axis=1)
    out = os.path.join(args.out_dir, f"{args.scene}_f{args.frame}_maskoverlay.png")
    Image.fromarray(combined).save(out)
    print("saved:", out)
    print("Left = real, right = real with chunk mask in red.")
    print("The red region should sit on the room area inside the cube.")


if __name__ == "__main__":
    main()