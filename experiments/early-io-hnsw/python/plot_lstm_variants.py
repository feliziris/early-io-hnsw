"""Plot baseline, two-layer and one-layer LSTM in one grouped breakdown PNG."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ("baseline", "lstm_2layer", "lstm_1layer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50])
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if len(set(args.k)) != len(args.k) or any(k <= 0 for k in args.k):
        parser.error("--k must contain distinct positive values")
    rows = {}
    with args.summary.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = int(row["k"]), row["mode"]
            if key in rows:
                parser.error(f"Duplicate summary entry: {key}")
            rows[key] = row
    data, counts = [], []
    for k in args.k:
        for mode in MODES:
            row = rows[k, mode]
            values = tuple(float(row[key]) for key in ("search_ms", "io_wait_ms", "e2e_ms"))
            if not all(math.isfinite(value) and value >= 0 for value in values):
                parser.error(f"Invalid latency: {(k, mode)}")
            if not math.isclose(values[0] + values[1], values[2], abs_tol=1e-6):
                parser.error(f"Search + I/O wait differs from E2E: {(k, mode)}")
            data.append(values)
            counts.append(int(row["queries"]))
    if len(set(counts)) != 1 or min(counts) <= 0:
        parser.error("All conditions must have the same positive query count")

    positions = [4 * group + offset for group in range(len(args.k)) for offset in range(3)]
    search, wait, totals = zip(*data)
    largest = max(totals)
    fig, ax = plt.subplots(figsize=(max(9, 5 * len(args.k)), 6))
    ax.bar(positions, search, width=.72, color="#2563eb", label="Search / inference phase")
    ax.bar(positions, wait, bottom=search, width=.72, color="#f97316", label="Final I/O wait")
    for x, (s, w, t) in zip(positions, data):
        if s > largest * .08:
            ax.text(x, s / 2, f"{s:.3f} ms", ha="center", va="center", color="white", fontsize=9)
        ax.annotate(f"Total: {t:.3f} ms\nI/O wait: {w:.3f} ms", (x, s + w),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(positions, ["Baseline", "2-layer LSTM", "1-layer LSTM"] * len(args.k))
    for group, k in enumerate(args.k):
        ax.text(4 * group + 1, -.11, f"k = {k}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean latency per query (ms)")
    ax.set_ylim(0, max(largest * 1.25, .001))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.14), ncol=2, frameon=False)
    fig.subplots_adjust(left=.065, right=.99, bottom=.19, top=.87)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
