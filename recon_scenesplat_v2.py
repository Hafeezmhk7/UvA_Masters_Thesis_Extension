"""
Encode interior_gs (.npy) scenes through the GaussianGPT VQ-VAE, decode them
back, and write INRIA-style .ply files for both the voxel-collapsed input and
the reconstruction, so they can be compared in any 3DGS viewer (SuperSplat etc).

Full scenes are ~0.9M-2M gaussians, so each scene is tiled into ~4.8 m chunks
(see scenesplat_npy.chunk_scene) and every chunk is reconstructed separately.

Place scenesplat_npy.py and this file in the GaussianGPT repo root. Run from
there, e.g.:

    python recon_scenesplat.py \
        --data-root /scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/test \
        --checkpoint checkpoints/vqvae_both.ckpt \
        --output-dir /scratch-shared/mkhan4/gaussian_world/ggpt_recon \
        --max-scenes 5
"""

import argparse
import os

import torch

from conf.dataclasses import GaussianFeatures, PCKeys
from data.common import collate_fn
from data.photoshape import save_inria_ply
from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import split_batch_dict

from scenesplat_npy import (
    chunk_scene,
    collapse_by_opacity,
    list_scenes,
    load_scenesplat_scene,
)

torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True, help="vqvae_both.ckpt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--chunk-m", type=float, default=4.8,
                   help="Chunk width in metres for tiling the scene in x-y.")
    p.add_argument("--voxel-select", choices=["random", "opacity"],
                   default="random",
                   help="How to pick one gaussian per 2.5cm voxel. "
                        "random = leave it to the model (default); "
                        "opacity = pre-collapse to highest-opacity per voxel.")
    p.add_argument("--skip-input-ply", action="store_true")
    return p.parse_args()


def payload_from_points(points, batch_idx=0):
    mask = points[PCKeys.BATCH] == batch_idx
    return {
        "coords": points[GaussianFeatures.COORDS][mask].cpu(),
        "sh0": points[GaussianFeatures.SH0][mask].cpu(),
        "opacities": points[GaussianFeatures.OPACITIES][mask].cpu(),
        "scales": points[GaussianFeatures.SCALES][mask].cpu(),
        "quats": points[GaussianFeatures.QUATS][mask].cpu(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    scenes = list_scenes(args.data_root)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    print(f"Found {len(scenes)} scenes to process.")

    ae = GaussianVQVAE.load_from_checkpoint(
        checkpoint_path=args.checkpoint, training_config=None
    )
    ae = ae.to(device).eval()

    with torch.inference_mode():
        for si, scene_id in enumerate(scenes):
            scene = load_scenesplat_scene(os.path.join(args.data_root, scene_id))
            n_scene = scene["coords"].shape[0]
            print(f"[{si + 1}/{len(scenes)}] {scene_id}: {n_scene} gaussians")

            for chunk_id, chunk in chunk_scene(scene, chunk_m=args.chunk_m):
                try:
                    if args.voxel_select == "opacity":
                        chunk = collapse_by_opacity(chunk)
                    item = collate_fn([chunk])
                    points, _ = split_batch_dict(item, device=device)

                    tok = ae.tokenize(points, sort_latents=None)
                    n_tokens = tok["coords"].shape[0]
                    if n_tokens == 0:
                        continue

                    recon = ae.decode(
                        tok["coords"].to(device), tok["feature_ids"].to(device)
                    )
                    recon_payload = payload_from_points(recon)
                    n_out = recon_payload["coords"].shape[0]

                    stem = f"{scene_id}_{chunk_id}_{args.voxel_select}"
                    save_inria_ply(
                        output_path=os.path.join(
                            args.output_dir, f"{stem}_recon.ply"
                        ),
                        **recon_payload,
                    )
                    torch.save(
                        recon_payload,
                        os.path.join(args.output_dir, f"{stem}_recon.pt"),
                    )
                    if not args.skip_input_ply:
                        save_inria_ply(
                            output_path=os.path.join(
                                args.output_dir, f"{stem}_input.ply"
                            ),
                            coords=chunk["coords"],
                            sh0=chunk["sh0"],
                            opacities=chunk["opacities"],
                            scales=chunk["scales"],
                            quats=chunk["quats"],
                        )

                    print(f"    {chunk_id}: in={chunk['coords'].shape[0]} "
                          f"tokens={n_tokens} out={n_out}")
                except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                    print(f"    {chunk_id}: OOM, skipping ({e})")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    {chunk_id}: FAILED ({e})")


if __name__ == "__main__":
    main()