"""
Poll the TensorBoard event file for a running training job and print the
current metrics to stdout every N seconds. Run in the background alongside
train_ae.py so the SLURM .out file shows live-updating train/val metrics
(the TQDM progress bar alone doesn't print metric values).

Usage:
    python poll_metrics.py --exp scenesplat_finetune_v3 --interval 120
"""

import argparse
import glob
import time


def latest_version(exp):
    vers = sorted(glob.glob(f"logs/{exp}/lightning_logs/version_*"),
                  key=lambda p: int(p.rsplit("_", 1)[1]))
    return vers[-1] if vers else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--interval", type=int, default=120,
                    help="Seconds between metric dumps.")
    args = ap.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    want = ["loss/train_epoch", "loss/val", "metrics/psnr", "metrics/ssim",
            "metrics/lpips", "loss_recon/train_vgg_loss",
            "loss_recon/val_l1_loss", "coords/train_accuracy",
            "loss_recon/train_mask_coverage", "vq/usage_pct"]

    while True:
        d = latest_version(args.exp)
        if d is None:
            print("[metrics] no log dir yet", flush=True)
            time.sleep(args.interval)
            continue
        try:
            ea = EventAccumulator(d)
            ea.Reload()
            tags = set(ea.Tags().get("scalars", []))
            line_parts = []
            step = None
            for t in want:
                if t in tags:
                    ev = ea.Scalars(t)
                    if ev:
                        step = ev[-1].step
                        short = t.split("/")[-1]
                        line_parts.append(f"{short}={ev[-1].value:.4f}")
            stamp = time.strftime("%H:%M:%S")
            if line_parts:
                print(f"[metrics {stamp} step {step}] " + "  ".join(line_parts),
                      flush=True)
            else:
                print(f"[metrics {stamp}] waiting for scalars...", flush=True)
        except Exception as e:
            print(f"[metrics] read error: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()