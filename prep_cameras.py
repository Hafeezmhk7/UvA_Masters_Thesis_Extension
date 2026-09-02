"""
Step 1 of fine-tuning prep: convert each scene's lang_feat_selected_imgs.json
into a transforms_train.json in the exact format the repo's
readCamerasFromTransforms (data/vfront.py) expects, and extract the images at
the target resolution.

Two subtleties this handles:

1. Pose convention. Yue's transform_matrix is OpenCV camera-to-world (y-down,
   z-forward = COLMAP). But readCamerasFromTransforms does `c2w[:3,1:3] *= -1`,
   which assumes OpenGL/Blender input (y-up, z-back) and flips it to COLMAP.
   Feeding OpenCV poses straight in would double-flip. So we pre-apply the same
   y/z flip here; the reader's flip then cancels it and recovers the true pose.
   (Validated separately: the un-flipped pose renders aligned via render_core,
   so cancelling the reader's flip reproduces that alignment.)

2. Intrinsics resolution. The stored fl_x=739 / cx,cy=1280,720 are for the full
   2560x1440. We target 960x540 (the `resize`), so we scale all intrinsics by
   960/2560 = 0.375 -> fl=277.1, and write a CENTERED principal point
   (cx=480, cy=270), which the reader uses since the path has no
   vfront/photoshape/3dfront keyword.

Output layout (per scene), under --out-root:
    <scene>/transforms_train.json
    <scene>/images/000000.png ...   (resized to 960x540)

Run (CPU is fine, no GPU needed):
    python prep_cameras.py --scenes 0201_840151 0202_840156
    python prep_cameras.py --all
"""

import argparse
import json
import os
import zipfile

import numpy as np
from PIL import Image

CAM_ROOT = "/scratch-shared/mkhan4/gaussian_world/camera_data/scenes"
OUT_ROOT_DEFAULT = "/scratch-shared/mkhan4/gaussian_world/finetune_data"

# cancels the reader's built-in c2w[:3,1:3] *= -1
YZ_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Specific scene ids. Omit with --all for everything.")
    p.add_argument("--all", action="store_true",
                   help="Process every scene found under CAM_ROOT.")
    p.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    p.add_argument("--target-w", type=int, default=960)
    p.add_argument("--target-h", type=int, default=540)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Keep every Nth frame (1=all). Use ~5 to subsample.")
    p.add_argument("--extract-images", action="store_true", default=True,
                   help="Unzip+resize images (needed for the render loss).")
    p.add_argument("--no-extract-images", dest="extract_images",
                   action="store_false")
    return p.parse_args()


def convert_scene(scene, out_root, tw, th, extract_images, frame_stride=1):
    cam_dir = os.path.join(CAM_ROOT, scene)
    meta = json.load(open(os.path.join(cam_dir, "lang_feat_selected_imgs.json")))

    full_w = int(meta["w"])                    # 2560
    scale = tw / full_w                        # 960/2560 = 0.375
    fl_x = float(meta["fl_x"]) * scale          # 739 * 0.375 = 277.1
    fl_y = float(meta["fl_y"]) * scale
    cx = tw / 2.0                               # centered: 480
    cy = th / 2.0                               # centered: 270

    out_dir = os.path.join(out_root, scene)
    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    frames_out = []
    n_bad = 0

    # keep the zip OPEN across the whole frame loop
    zf = zipfile.ZipFile(os.path.join(cam_dir, "images.zip")) if extract_images else None
    names = {}
    if zf is not None:
        names = {os.path.basename(n): n for n in zf.namelist()
                 if n.lower().endswith((".jpg", ".jpeg", ".png"))}

    try:
        for _fi, fr in enumerate(meta["frames"]):
            if _fi % frame_stride != 0:
                continue
            if fr.get("is_bad"):
                n_bad += 1
                continue
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            c2w = c2w @ YZ_FLIP                     # pre-cancel the reader's flip
            base = os.path.basename(fr["file_path"])  # e.g. 000000.jpg
            stem = os.path.splitext(base)[0]
            out_name = f"{stem}.png"

            if zf is not None and base in names:
                with zf.open(names[base]) as fh:
                    Image.open(fh).convert("RGB").resize((tw, th)).save(
                        os.path.join(img_out, out_name))

            frames_out.append({
                "file_path": f"images/{out_name}",
                "transform_matrix": c2w.tolist(),
            })
    finally:
        if zf is not None:
            zf.close()

    transforms = {
        "fl_x": fl_x, "fl_y": fl_y,
        "cx": cx, "cy": cy,
        "w": tw, "h": th,
        "frames": frames_out,
    }
    with open(os.path.join(out_dir, "transforms_train.json"), "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"{scene}: {len(frames_out)} frames "
          f"({n_bad} bad skipped), fl={fl_x:.1f} cx={cx} cy={cy} @ {tw}x{th}")


def main():
    args = parse_args()
    if args.all:
        scenes = sorted(d for d in os.listdir(CAM_ROOT)
                        if os.path.isdir(os.path.join(CAM_ROOT, d)))
    else:
        scenes = args.scenes or []
    if not scenes:
        print("No scenes given. Use --scenes <ids> or --all.")
        return
    print(f"Converting {len(scenes)} scenes -> {args.out_root}")
    for s in scenes:
        convert_scene(s, args.out_root, args.target_w, args.target_h,
                      args.extract_images, args.frame_stride)


if __name__ == "__main__":
    main()