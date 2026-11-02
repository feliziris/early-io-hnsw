"""Compare two independently measured LSTM-layer experiments in two PNG panels."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(root, topks):
    rows = {}
    with (root / "summary_ms.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (int(row["k"]), row["mode"])
            if key in rows:
                raise ValueError(f"Duplicate row in {root}: {key}")
            rows[key] = row
    data = []
    counts = []
    for k in topks:
        for mode in ("baseline", "prefetch"):
            row = rows[(k, mode)]
            values = tuple(float(row[name]) for name in ("search_ms", "io_wait_ms", "e2e_ms"))
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f"Invalid latency in {root}: {(k, mode)}")
            if not math.isclose(values[0] + values[1], values[2], abs_tol=1e-6):
                raise ValueError(f"Search + wait differs from E2E in {root}: {(k, mode)}")
            data.append(values)
            counts.append(int(row["queries"]))
    return data, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-layer-root", type=Path, required=True)
    parser.add_argument("--one-layer-root", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if any(k <= 0 for k in args.k) or len(set(args.k)) != len(args.k):
        parser.error("--k values must be positive and unique")
    two, two_counts = load_summary(args.two_layer_root, args.k)
    one, one_counts = load_summary(args.one_layer_root, args.k)
    if len(set(two_counts + one_counts)) != 1 or min(two_counts + one_counts) <= 0:
        parser.error("All conditions must have the same positive query count")

    positions = [3 * group + offset for group in range(len(args.k)) for offset in (0, 1)]
    largest = max(row[2] for row in two + one)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(max(14, 7 * len(args.k)), 6.2))
    for ax, data, label in zip(axes, (two, one), ("(a) 2-layer LSTM", "(b) 1-layer LSTM")):
        search, wait, _ = zip(*data)
        ax.bar(positions, search, width=.65, color="#2563eb", label="Search / inference phase")
        ax.bar(positions, wait, bottom=search, width=.65, color="#f97316", label="Final I/O wait")
        for x, (s, w, t) in zip(positions, data):
            if s > largest * .08:
                ax.text(x, s / 2, f"{s:.3f} ms", ha="center", va="center", color="white", fontsize=10)
            ax.annotate(f"Total: {t:.3f} ms\nI/O wait: {w:.3f} ms", (x, s + w),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(positions, ["Baseline", "Prefetch"] * len(args.k))
        for group, k in enumerate(args.k):
            ax.text(3 * group + .5, -.10, f"k = {k}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontweight="bold")
        ax.text(.5, -.23, label, transform=ax.transAxes, ha="center", va="top", fontsize=13)
        ax.set_ylim(0, max(largest * 1.25, .001))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean latency per query (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.055, right=.99, bottom=.24, top=.88, wspace=.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"2-layer source: {args.two_layer_root / 'summary_ms.tsv'}")
    print(f"1-layer source: {args.one_layer_root / 'summary_ms.tsv'}")
    print(f"Queries per condition: {two_counts[0]}; independent baselines retained")
    print(args.output)


if __name__ == "__main__":
    main()
