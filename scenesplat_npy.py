"""
Load SceneSplat-style .npy scenes (interior_gs) into the pre-activation
Gaussian dict that the GaussianGPT VQ-VAE expects.

Each scene folder holds:
    coord.npy    (N,3) float   positions in metres
    scale.npy    (N,3) float   scales in metres (LINEAR / activated, all >= 0)
    quat.npy     (N,4) float   unit quaternions
    opacity.npy  (N,)  float   opacity in [0,1] (activated)
    color.npy    (N,3) uint8   RGB in [0,255]

GaussianGPT's renderer applies exp() to scales and sigmoid() to opacities,
and treats sh0 as SH DC coefficients. So we invert those here:
    scales   -> log(scale)
    opacity  -> logit(opacity)
    color    -> (rgb/255 - 0.5) / 0.28209479   (RGB -> SH DC)

Returns everything as float32 torch tensors under the keys the VQ-VAE reads:
    coords, sh0, opacities, scales, quats
"""

import os

import numpy as np
import torch

SH_C0 = 0.28209479177387814  # DC term of the SH basis


def list_scenes(data_root):
    return sorted(
        d
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and os.path.exists(os.path.join(data_root, d, "coord.npy"))
    )


def load_scenesplat_scene(scene_dir, eps=1e-6):
    coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
    scale = np.load(os.path.join(scene_dir, "scale.npy")).astype(np.float32)
    quat = np.load(os.path.join(scene_dir, "quat.npy")).astype(np.float32)
    opacity = np.load(os.path.join(scene_dir, "opacity.npy")).astype(np.float32)
    color = np.load(os.path.join(scene_dir, "color.npy")).astype(np.float32)

    # invert scale activation: metres -> log
    scale = np.log(np.clip(scale, eps, None))

    # invert opacity activation: [0,1] -> logit, clamped to avoid +/- inf
    opacity = np.clip(opacity, eps, 1.0 - eps)
    opacity = np.log(opacity / (1.0 - opacity))

    # RGB [0,255] -> SH DC coefficient
    sh0 = (color / 255.0 - 0.5) / SH_C0

    # renormalize quats (float16 storage can drift slightly)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + eps)

    return {
        "coords": torch.from_numpy(coord),
        "sh0": torch.from_numpy(sh0),
        "opacities": torch.from_numpy(opacity),
        "scales": torch.from_numpy(scale),
        "quats": torch.from_numpy(quat),
    }


def chunk_scene(scene, chunk_m=4.8, min_pts=2000):
    """
    Split a scene into axis-aligned chunks in the x-y plane, each chunk_m
    metres wide (z is left whole). GaussianGPT trains on ~4.8 m chunks, and
    full interior_gs scenes (up to ~2M gaussians) won't tokenize in one shot,
    so we tile them. Yields (chunk_id, chunk_dict). z is kept intact because
    room height already fits the model's chunk extent.
    """
    coords = scene["coords"]
    xy = coords[:, :2]
    mins = xy.min(dim=0).values
    ij = ((xy - mins) / chunk_m).floor().long()  # (N,2) tile index

    keys = ij[:, 0] * 100000 + ij[:, 1]
    for key in torch.unique(keys):
        mask = keys == key
        if mask.sum() < min_pts:
            continue
        gx = int(key.item() // 100000)
        gy = int(key.item() % 100000)
        chunk = {k: v[mask] for k, v in scene.items()}
        yield f"x{gx}_y{gy}", chunk