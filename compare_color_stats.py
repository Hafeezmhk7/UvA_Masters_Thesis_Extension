"""
Automates the gap-closure comparison between two color_stats.py runs (e.g.
1-token baseline vs 4-token RVQ4), instead of transcribing numbers by hand.
Also removes the threshold-flip-flop problem in color_stats.py's old verdict:
this reports a continuous percent-of-gap-closed, so a run that moves from
0.842 to 0.873 doesn't get miscategorized as a qualitative change just
because it crossed some fixed cutoff.

Usage:
    python color_stats.py --scenes ... --checkpoint <baseline.ckpt> \
        --label baseline_1tok --save-json stats_baseline.json
    python color_stats.py --scenes ... --checkpoint <rvq4.ckpt> \
        --label rvq4_epoch9 --save-json stats_rvq4.json
    python compare_color_stats.py stats_baseline.json stats_rvq4.json
"""

import argparse
import json


def pct_gap_closed(ratio_before: float, ratio_after: float) -> float:
    """% of the (before) gap-to-parity that (after) closes. Positive = better,
    negative = the gap widened. Undefined (returns None) if there was no gap
    to close (ratio_before >= 1.0).
    """
    gap_before = 1.0 - ratio_before
    if gap_before <= 1e-9:
        return None
    return 100.0 * (ratio_after - ratio_before) / gap_before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before", help="JSON from the earlier/weaker run (e.g. baseline).")
    ap.add_argument("after", help="JSON from the later/stronger run (e.g. RVQ4).")
    args = ap.parse_args()

    with open(args.before) as f:
        a = json.load(f)
    with open(args.after) as f:
        b = json.load(f)

    label_a = a.get("label") or args.before
    label_b = b.get("label") or args.after

    print("=" * 70)
    print(f"COMPARISON: {label_a}  ->  {label_b}")
    print("=" * 70)

    rows = [
        ("mean std ratio", a["std_ratio_mean"], b["std_ratio_mean"]),
        ("saturation ratio", a["saturation_ratio"], b["saturation_ratio"]),
    ]
    for i, ch in enumerate("RGB"):
        rows.append((f"{ch} std ratio", a["std_ratio_per_channel"][i], b["std_ratio_per_channel"][i]))

    print(f"{'metric':20s} {label_a[:14]:>14s} {label_b[:14]:>14s} {'gap closed':>12s}")
    for name, ra, rb in rows:
        pct = pct_gap_closed(ra, rb)
        pct_str = f"{pct:+.1f}%" if pct is not None else "n/a (no gap)"
        print(f"{name:20s} {ra:14.3f} {rb:14.3f} {pct_str:>12s}")

    max_shift_a = max(abs(x) for x in a["mean_shift_per_channel"])
    max_shift_b = max(abs(x) for x in b["mean_shift_per_channel"])
    print(f"{'max mean shift':20s} {max_shift_a:14.4f} {max_shift_b:14.4f}")

    print("\nInterpretation notes (read the numbers, not a fixed cutoff):")
    var_pct = pct_gap_closed(a["std_ratio_mean"], b["std_ratio_mean"])
    sat_pct = pct_gap_closed(a["saturation_ratio"], b["saturation_ratio"])
    if var_pct is not None and sat_pct is not None:
        if var_pct > 5 and abs(sat_pct) < 5:
            print(
                f"  -> variance/local-contrast gap closed by {var_pct:.0f}% while saturation "
                f"barely moved ({sat_pct:+.0f}%). This dissociation suggests capacity buys back "
                "spatial detail but not saturation -- treat them as two separate mechanisms "
                "needing separate fixes (more tokens/capacity for detail, a generative decoder "
                "or a dedicated saturation term for the rest) until a later checkpoint shows "
                "otherwise."
            )
        elif var_pct > 5 and sat_pct > 5:
            print(
                f"  -> both variance ({var_pct:.0f}%) and saturation ({sat_pct:.0f}%) gaps closed "
                "together -- capacity may be addressing both, or both are still converging with "
                "training (check whether the two checkpoints are at comparable epochs before "
                "concluding capacity is the shared cause)."
            )
        else:
            print("  -> no large, consistent movement on either metric between these two runs.")
    if abs(max_shift_a - max_shift_b) < 1e-4:
        print(
            f"  -> max mean shift is essentially unchanged ({max_shift_a:.4f} vs {max_shift_b:.4f}): "
            "whatever causes the small mean shift is not affected by this change, consistent with "
            "it being a separate (likely render/init-level, not capacity-level) effect."
        )


if __name__ == "__main__":
    main()
