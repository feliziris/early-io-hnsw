#!/usr/bin/env python3
"""Compare prefetch latency breakdowns at thresholds 0.5 and 0.8."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for topk in (10, 20, 50):
        parser.add_argument(f"--threshold05-k{topk}", required=True, type=Path)
        parser.add_argument(f"--threshold08-k{topk}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_prefetch(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if row["mode"] == "on"]
    if len(matches) != 1:
        raise ValueError(f"{path} must contain exactly one prefetch-on row")
    row = matches[0]
    return {
        "search": float(row["search_mean_us"]) / 1000.0,
        "wait": float(row["data_ready_wait_mean_us"]) / 1000.0,
        "total": float(row["end_to_end_mean_us"]) / 1000.0,
    }


def main() -> None:
    args = parse_args()
    topks = (10, 20, 50)
    thresholds = ("0.5", "0.8")
    values = {
        topk: {
            "0.5": load_prefetch(getattr(args, f"threshold05_k{topk}")),
            "0.8": load_prefetch(getattr(args, f"threshold08_k{topk}")),
        }
        for topk in topks
    }

    colors = {"search": "#2563EB", "wait": "#F97316"}
    group_centers = {10: 0.0, 20: 4.0, 50: 8.0}
    offsets = {"0.5": -0.5, "0.8": 0.5}
    bar_width = 0.78

    fig, ax = plt.subplots(figsize=(14, 8))
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for topk in topks:
        for threshold in thresholds:
            position = group_centers[topk] + offsets[threshold]
            search = values[topk][threshold]["search"]
            wait = values[topk][threshold]["wait"]
            total = values[topk][threshold]["total"]

            ax.bar(
                position,
                search,
                width=bar_width,
                color=colors["search"],
                label="HNSW search + inference"
                if topk == 10 and threshold == "0.5"
                else None,
            )
            ax.bar(
                position,
                wait,
                width=bar_width,
                bottom=search,
                color=colors["wait"],
                label="Raw-data ready wait"
                if topk == 10 and threshold == "0.5"
                else None,
            )

            ax.text(
                position,
                search / 2,
                f"{search:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
            annotation_x = -10 if threshold == "0.5" else 10
            annotation_alignment = "right" if threshold == "0.5" else "left"
            ax.annotate(
                f"I/O wait {wait:.3f} ms",
                xy=(position, total),
                xytext=(annotation_x, 19),
                textcoords="offset points",
                ha=annotation_alignment,
                va="bottom",
                color=colors["wait"],
                fontsize=9,
                fontweight="bold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": colors["wait"],
                    "linewidth": 1.0,
                },
            )
            ax.annotate(
                f"Total {total:.3f} ms",
                xy=(position, total),
                xytext=(annotation_x, 5),
                textcoords="offset points",
                ha=annotation_alignment,
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            tick_positions.append(position)
            tick_labels.append(f"Threshold {threshold}")

    ax.set_xticks(tick_positions, tick_labels)
    for topk in topks:
        ax.text(
            group_centers[topk],
            -0.105,
            f"k={topk}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean latency (ms/query)", fontsize=12)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(bottom=0.17, top=0.98)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
