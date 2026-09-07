"""
Dormancy / capacity-curve check for a trained ResidualLFQ (num_tokens>1)
checkpoint: re-evaluates the SAME trained model with only the first
`n_active` of its residual quantizer layers contributing, for n_active in
1..num_tokens.

Why this is the right test (not just re-running with a smaller num_tokens
config): ResidualLFQ's own source (vector_quantize_pytorch.residual_lfq)
builds each successive layer's codebook_scale as 2**-ind, i.e. layer 0 is the
coarse code and layers 1.. are progressively finer, smaller residual
corrections -- the model was trained to actually use this coarse-to-fine
structure. get_output_from_indices() does:
    codes = get_codes_from_indices(indices)   # per-layer codes
    codes_summed = reduce(codes, 'q ... -> ...', 'sum')
    return project_out(codes_summed)
so truncating to n_active layers means summing only the first n_active
layers' codes before project_out -- NOT zeroing/clamping indices (an LFQ
code is +-1, there is no "null" index that contributes zero).

Interpretation:
  - If n_active=1 reproduces ~baseline (1-token) PSNR/LPIPS and quality rises
    steadily through n_active=4, the residual layers are genuinely carrying
    the +0.6 dB gain -- the capacity story holds, and this IS your
    reconstruction-fidelity-vs-sequence-length curve for the GPT tradeoff.
  - If n_active=1 already sits near the full 4-token number, the RVQ4 gain
    over the 1-token baseline model came from somewhere else (different init,
    different effective training trajectory) and the extra tokens are mostly
    dormant.

Run on a GPU node from the repo root:
    python rvq_truncation_check.py \
        --scenes 0390_841511 0391_841500 0392_841498 \
        --checkpoint logs/scenesplat_ft_rvq4/lightning_logs/version_X/checkpoints/<ckpt> \
        --cams-per-scene 8
"""

import argparse
import contextlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval_quant import FT_ROOT, GS_ROOT, psnr, reconstruct_full, render_from, try_lpips
from model.gaussian_vqvae import GaussianVQVAE
from scenesplat_npy import load_scenesplat_scene


def _find_quantizer_axis(codes: torch.Tensor, num_quantizers: int) -> int:
    """Defensive lookup: which axis of `codes` indexes the residual layers.

    Written against vector_quantize_pytorch's current ResidualLFQ, where
    get_codes_from_indices returns shape (num_quantizers, ..., codebook_dim).
    If your installed version differs, this raises with the actual shape so
    you can adjust the axis by hand instead of silently computing garbage.
    """
    matches = [i for i, s in enumerate(codes.shape) if s == num_quantizers]
    if not matches:
        raise RuntimeError(
            f"Could not find an axis of size num_quantizers={num_quantizers} in "
            f"codes.shape={tuple(codes.shape)}. The installed vector_quantize_pytorch "
            "version's get_codes_from_indices() layout differs from what this script "
            "assumes -- print codes.shape yourself and adjust `dim=` below."
        )
    return matches[0]


@contextlib.contextmanager
def truncated_residual(sparse_quantizer, n_active: int):
    """Monkeypatch SparseLatentVectorQuantizer.get_output_from_idxs so decoding
    only sums the first `n_active` ResidualLFQ layers' codes.
    """
    num_quantizers = sparse_quantizer.num_tokens
    assert 1 <= n_active <= num_quantizers
    orig = sparse_quantizer.get_output_from_idxs

    def patched(idxs: torch.Tensor) -> torch.Tensor:
        if n_active == num_quantizers:
            return orig(idxs)  # exact fallback: must reproduce the untruncated result
        rvq = sparse_quantizer.vq
        codes = rvq.get_codes_from_indices(idxs)  # per-layer codes, one axis == num_quantizers
        axis = _find_quantizer_axis(codes, num_quantizers)
        codes = codes.narrow(axis, 0, n_active).sum(dim=axis)
        return rvq.project_out(codes)

    sparse_quantizer.get_output_from_idxs = patched
    try:
        yield
    finally:
        sparse_quantizer.get_output_from_idxs = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True, help="Trained RVQ (num_tokens>1) checkpoint.")
    ap.add_argument("--cams-per-scene", type=int, default=8)
    ap.add_argument("--chunk-m", type=float, default=4.0)
    args = ap.parse_args()

    device = torch.device("cuda")
    lpips_fn = try_lpips()

    print("loading model...")
    ae = GaussianVQVAE.load_from_checkpoint(args.checkpoint, training_config=None).to(device).eval()
    sparse_quantizer = ae.autoencoder.vq
    num_quantizers = sparse_quantizer.num_tokens
    if num_quantizers <= 1:
        raise ValueError(
            f"Checkpoint has num_tokens={num_quantizers}; this script only makes sense "
            "for a trained ResidualLFQ (num_tokens > 1) checkpoint, e.g. your RVQ4 run."
        )
    print(f"checkpoint has num_tokens (residual layers) = {num_quantizers}")

    # cache per-scene GT + camera setup once; reconstruct per n_active below
    scenes_data = []
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
        g_gt = {k: scene[k] for k in ["coords", "sh0", "opacities", "scales", "quats"]}
        frames = d["frames"]
        idxs = np.linspace(0, len(frames) - 1, args.cams_per_scene).astype(int)
        scenes_data.append((scene_id, scene, g_gt, d, W, H, K, idxs))

    results = {}
    for n_active in range(1, num_quantizers + 1):
        agg_psnr, agg_lpips = [], []
        with truncated_residual(sparse_quantizer, n_active):
            for scene_id, scene, g_gt, d, W, H, K, cam_idxs in scenes_data:
                print(f"[n_active={n_active}] {scene_id}: reconstructing...")
                g_re = reconstruct_full(ae, scene, device, args.chunk_m)
                if g_re is None:
                    continue
                for fi in cam_idxs:
                    c2w = np.array(d["frames"][fi]["transform_matrix"], dtype=np.float64).copy()
                    c2w[:3, 1:3] *= -1.0
                    view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device).unsqueeze(0)
                    gt = render_from(g_gt, view, K, W, H, device)
                    im = render_from(g_re, view, K, W, H, device)
                    p = psnr(im, gt)
                    if p is not None:
                        agg_psnr.append(p)
                    if lpips_fn is not None:
                        with torch.inference_mode():
                            lp = lpips_fn(im.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()
                        agg_lpips.append(lp)
        pm = float(np.mean([x for x in agg_psnr if np.isfinite(x)])) if agg_psnr else float("nan")
        lm = float(np.mean(agg_lpips)) if agg_lpips else float("nan")
        results[n_active] = (pm, lm, len(agg_psnr))
        print(f"  -> n_active={n_active}: PSNR={pm:.3f} (n={len(agg_psnr)})  LPIPS={lm:.4f}")

    print("\n" + "=" * 60)
    print("CAPACITY CURVE (same trained RVQ checkpoint, truncated at inference)")
    print("=" * 60)
    for n_active, (pm, lm, n) in results.items():
        print(f"n_active={n_active}/{num_quantizers}   PSNR={pm:.3f} (n={n})   LPIPS={lm:.4f}")

    p1 = results[1][0]
    pfull = results[num_quantizers][0]
    if np.isfinite(p1) and np.isfinite(pfull):
        gained = pfull - p1
        print(f"\nPSNR gained from layer 1 alone -> all {num_quantizers} layers: {gained:+.3f} dB")
        print(
            "If this is close to your measured (RVQ4 full) - (1-token baseline) gap, "
            "the residual layers are genuinely carrying the improvement. If n_active=1 "
            "here already sits near the full-depth number, the earlier +0.6 dB was mostly "
            "incidental (init/trajectory), not capacity."
        )


if __name__ == "__main__":
    main()
