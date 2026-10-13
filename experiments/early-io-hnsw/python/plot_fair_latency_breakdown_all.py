#!/usr/bin/env python3
"""Plot worker-matched latency breakdowns for k=10, 20, and 50."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument(
        "--paired-summary",
        type=Path,
        help="paired_mean.tsv containing baseline and oracle/replay rows",
    )
    for topk in (10, 20, 50):
        parser.add_argument(f"--summary-k{topk}", type=Path)
    parser.add_argument("--query-count", type=int, default=2000)
    parser.add_argument(
        "--subtitle",
        help="optional subtitle; defaults to the query/worker configuration",
    )
    parser.add_argument(
        "--no-subtitle",
        action="store_true",
        help="omit the subtitle entirely",
    )
    parser.add_argument(
        "--oracle-summary",
        type=Path,
        help="optional oracle_summary.tsv to add an Oracle bar to each k group",
    )
    parser.add_argument(
        "--exclude-prefetch",
        action="store_true",
        help="omit the learned Prefetch bars (requires --oracle-summary)",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if set(rows) != {"off", "on"}:
        raise ValueError(f"{path} must contain exactly off and on rows")
    return {
        mode: {
            "search": float(rows[mode]["search_mean_us"]) / 1000.0,
            "wait": float(rows[mode]["data_ready_wait_mean_us"]) / 1000.0,
            "total": float(rows[mode]["end_to_end_mean_us"]) / 1000.0,
        }
        for mode in ("off", "on")
    }


def load_oracle_summary(path: Path) -> dict[int, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    values = {
        int(row["topk"]): {
            "search": float(row["search_mean_us"]) / 1000.0,
            "wait": float(row["data_ready_wait_mean_us"]) / 1000.0,
            "total": float(row["end_to_end_mean_us"]) / 1000.0,
        }
        for row in rows
        if row["mode"] == "oracle"
    }
    if set(values) != {10, 20, 50}:
        raise ValueError(f"{path} must contain oracle rows for k=10, 20, and 50")
    return values


def load_paired_summary(
    path: Path,
) -> tuple[dict[int, dict[str, dict[str, float]]], str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    values: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        topk = int(row["topk"])
        mode = row["mode"]
        values.setdefault(topk, {})[mode] = {
            "search": float(row["search_mean_us"]) / 1000.0,
            "wait": float(row["data_ready_wait_mean_us"]) / 1000.0,
            "total": float(row["end_to_end_mean_us"]) / 1000.0,
        }
    comparison_modes = {
        mode for topk_values in values.values() for mode in topk_values
    } - {"baseline"}
    if len(comparison_modes) != 1:
        raise ValueError(f"{path} must contain one paired comparison mode")
    comparison_mode = next(iter(comparison_modes))
    expected_modes = {"baseline", comparison_mode}
    if set(values) != {10, 20, 50} or any(
        set(values[topk]) != expected_modes for topk in values
    ):
        raise ValueError(
            f"{path} must contain baseline/{comparison_mode} rows for "
            "k=10, 20, and 50"
        )
    return values, comparison_mode


def main() -> None:
    args = parse_args()
    topks = (10, 20, 50)
    if args.paired_summary:
        values, comparison_mode = load_paired_summary(args.paired_summary)
        modes = ("baseline", comparison_mode)
        comparison_labels = {
            "oracle": "Oracle",
            "replay": "LSTM replay\n(0 inference cost)",
        }
        labels = {
            "baseline": "Baseline",
            comparison_mode: comparison_labels.get(
                comparison_mode, comparison_mode.capitalize()
            ),
        }
    else:
        summary_paths = {
            topk: getattr(args, f"summary_k{topk}") for topk in topks
        }
        if any(path is None for path in summary_paths.values()):
            if args.result_root is None:
                raise ValueError(
                    "provide --paired-summary, --result-root, or all of "
                    "--summary-k10/20/50"
                )
            summary_paths = {
                topk: args.result_root / f"k{topk}" / "summary.tsv"
                for topk in topks
            }
        values = {
            topk: load_summary(summary_paths[topk])
            for topk in topks
        }
        modes = ("off", "on")
        labels = {"off": "Baseline", "on": "Prefetch"}
        if args.oracle_summary:
            oracle_values = load_oracle_summary(args.oracle_summary)
            for topk in topks:
                values[topk]["oracle"] = oracle_values[topk]
            modes = ("off", "on", "oracle")
            labels["oracle"] = "Oracle"
        if args.exclude_prefetch:
            if not args.oracle_summary:
                raise ValueError("--exclude-prefetch requires --oracle-summary")
            modes = ("off", "oracle")

    colors = {"search": "#2563EB", "wait": "#F97316"}
    group_centers = {10: 0.0, 20: 4.0, 50: 8.0}
    offsets = {
        mode: index - (len(modes) - 1) / 2 for index, mode in enumerate(modes)
    }
    bar_width = 0.78

    fig, ax = plt.subplots(figsize=(14, 8))
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for topk in topks:
        for mode in modes:
            position = group_centers[topk] + offsets[mode]
            search = values[topk][mode]["search"]
            wait = values[topk][mode]["wait"]
            total = values[topk][mode]["total"]
            ax.bar(
                position,
                search,
                width=bar_width,
                color=colors["search"],
                label=(
                    "HNSW search"
                    if args.exclude_prefetch or args.paired_summary
                    else "HNSW search (+ inference for Prefetch)"
                )
                if topk == 10 and mode == modes[0]
                else None,
            )
            ax.bar(
                position,
                wait,
                width=bar_width,
                bottom=search,
                color=colors["wait"],
                label="Raw-data ready wait"
                if topk == 10 and mode == modes[0]
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
            if wait >= 0.5:
                ax.text(
                    position,
                    search + wait / 2,
                    f"{wait:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.annotate(
                    f"I/O wait {wait:.3f} ms",
                    xy=(position, total),
                    xytext=(0, 19),
                    textcoords="offset points",
                    ha="center",
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
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            tick_positions.append(position)
            tick_labels.append(labels[mode])

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
    if not args.no_subtitle:
        ax.text(
            0.5,
            1.01,
            args.subtitle
            or f"{args.query_count:,} seed-42 test queries · efSearch 200 · "
            "4 I/O workers · batch size 1",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4B5563",
        )
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(bottom=0.17, top=0.93)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
