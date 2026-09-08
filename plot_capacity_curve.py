"""
Reconstruction fidelity vs quantizer capacity -- the capacity-curve figure:
does PSNR / color-variance-ratio keep climbing as num_tokens (residual LFQ
layers) increases, or does it saturate? Two points define a slope; three or
more tell you whether it's saturating, which is the number that decides
whether to keep pushing capacity or switch to a generative decoder.

Deliberately two side-by-side panels sharing one x-axis rather than a single
dual-axis plot: PSNR (dB) and the two ratio metrics (0-1, "gap to parity")
are different scales, and overlaying them behind two y-axes make slopes
visually incomparable and easy to misread. Small multiples avoid that.

num_tokens IS the GPT sequence-length cost per occupied voxel (one residual
token per layer), so the x-axis already carries that tradeoff -- no need for
a third axis or panel for it, just say so in the caption.

Points from a run that hadn't finished training (converged=no in the CSV)
are drawn hollow with an epoch annotation, since an early checkpoint at
higher capacity can look better than a converged checkpoint at lower
capacity for reasons unrelated to capacity -- don't let the figure imply
more confidence in those points than you have.

Input CSV columns: label, num_tokens, epoch, converged (yes/no), psnr,
std_ratio, saturation_ratio. Append a row per checkpoint as new runs finish
(8-token, the RVQ4 epoch-29 re-run, ...); no code changes needed.

Usage:
    python plot_capacity_curve.py capacity_curve_data.csv --out capacity_curve.png
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"  # categorical slot 1 -- variance ratio / PSNR
ORANGE = "#eb6834"  # categorical slot 2 -- saturation ratio
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dc"


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["num_tokens"] = int(r["num_tokens"])
        r["epoch"] = int(r["epoch"]) if r.get("epoch") else None
        r["converged"] = str(r.get("converged", "yes")).strip().lower() in ("yes", "true", "1")
        r["psnr"] = float(r["psnr"])
        r["std_ratio"] = float(r["std_ratio"])
        r["saturation_ratio"] = float(r["saturation_ratio"])
    return sorted(rows, key=lambda r: r["num_tokens"])


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def plot_series(ax, rows, key, color, label):
    x = [r["num_tokens"] for r in rows]
    y = [r[key] for r in rows]
    ax.plot(x, y, color=color, linewidth=2, zorder=2, label=label)
    filled = [r for r in rows if r["converged"]]
    hollow = [r for r in rows if not r["converged"]]
    if filled:
        ax.scatter([r["num_tokens"] for r in filled], [r[key] for r in filled],
                   s=64, color=color, edgecolor="white", linewidth=1.2, zorder=3)
    if hollow:
        ax.scatter([r["num_tokens"] for r in hollow], [r[key] for r in hollow],
                   s=64, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
    for r in hollow:
        ax.annotate(f"epoch {r['epoch']}", (r["num_tokens"], r[key]),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=7.5, color=TEXT_SECONDARY, style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="capacity_curve.png")
    args = ap.parse_args()

    rows = load_rows(args.csv_path)
    tokens = sorted({r["num_tokens"] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6), facecolor="#fcfcfb")

    plot_series(ax1, rows, "psnr", BLUE, "PSNR")
    ax1.set_xlabel("residual tokens per voxel (num_tokens)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Reconstruction fidelity", color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(tokens)
    ax1.set_xticklabels([str(t) for t in tokens])
    style_axis(ax1)

    plot_series(ax2, rows, "std_ratio", BLUE, "color-variance ratio")
    plot_series(ax2, rows, "saturation_ratio", ORANGE, "saturation ratio")
    ax2.axhline(1.0, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
    ax2.annotate("parity with GT", (tokens[0], 1.0), textcoords="offset points",
                 xytext=(0, 4), fontsize=7.5, color=TEXT_SECONDARY)
    ax2.set_xlabel("residual tokens per voxel (num_tokens)")
    ax2.set_ylabel("ratio to ground truth")
    ax2.set_title("Color-statistics gap to parity", color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(tokens)
    ax2.set_xticklabels([str(t) for t in tokens])
    ax2.set_ylim(0.6, 1.05)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=TEXT_PRIMARY)
    style_axis(ax2)

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(
        0.5, 0.02,
        "num_tokens is also the GPT sequence-length cost per occupied voxel "
        "(one residual token per layer)",
        color=TEXT_SECONDARY, fontsize=8.5, ha="center",
    )
    fig.savefig(args.out, dpi=300, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
