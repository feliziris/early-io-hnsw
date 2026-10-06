#!/usr/bin/env python3
"""Plot live HNSW search/raw-data-wait latency as stacked SVG bars."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topk", required=True, type=int)
    parser.add_argument("--query-count", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if set(rows) != {"off", "on"}:
        raise ValueError("summary must contain exactly prefetch off and on rows")

    values = {
        mode: {
            "search": float(rows[mode]["search_mean_us"]) / 1000.0,
            "wait": float(rows[mode]["data_ready_wait_mean_us"]) / 1000.0,
            "total": float(rows[mode]["end_to_end_mean_us"]) / 1000.0,
        }
        for mode in ("off", "on")
    }

    width, height = 760, 600
    left, right, top, bottom = 100, 55, 90, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    tick_step = 50.0
    maximum = max(item["total"] for item in values.values())
    y_max = max(tick_step, math.ceil(maximum / tick_step) * tick_step)

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / y_max)

    def escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:23px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}.axis{font-size:13px}.segment{font-size:12px;font-weight:700;fill:#ffffff}.total{font-size:13px;font-weight:700}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2}" y="34" text-anchor="middle">k={args.topk} End-to-End Latency Breakdown</text>',
        f'<text class="subtitle" x="{width / 2}" y="58" text-anchor="middle">{args.query_count:,} seed-42 test queries · batch size 1 · lower is better</text>',
    ]

    tick = 0.0
    while tick <= y_max + 1e-9:
        tick_y = y(tick)
        svg.append(
            f'<line x1="{left}" y1="{tick_y:.2f}" x2="{width-right}" y2="{tick_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="axis" x="{left-12}" y="{tick_y+4:.2f}" text-anchor="end">{tick:.0f}</text>'
        )
        tick += tick_step

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#374151" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#374151" stroke-width="1.5"/>',
            f'<text class="axis" x="25" y="{top + plot_height/2}" text-anchor="middle" transform="rotate(-90 25 {top + plot_height/2})">Mean latency (ms/query)</text>',
        ]
    )

    bar_width = 135
    centers = {"off": left + plot_width * 0.30, "on": left + plot_width * 0.70}
    mode_labels = {"off": "Baseline", "on": "Prefetch"}
    colors = {"search": "#2563eb", "wait": "#f97316"}
    for mode in ("off", "on"):
        x = centers[mode] - bar_width / 2
        search = values[mode]["search"]
        wait = values[mode]["wait"]
        total = values[mode]["total"]
        baseline_y = y(0.0)
        search_top = y(search)
        total_top = y(total)
        search_height = baseline_y - search_top
        wait_height = search_top - total_top
        svg.append(
            f'<rect x="{x:.2f}" y="{search_top:.2f}" width="{bar_width}" height="{search_height:.2f}" fill="{colors["search"]}"/>'
        )
        svg.append(
            f'<rect x="{x:.2f}" y="{total_top:.2f}" width="{bar_width}" height="{wait_height:.2f}" rx="3" fill="{colors["wait"]}"/>'
        )
        if search_height >= 18:
            svg.append(
                f'<text class="segment" x="{centers[mode]:.2f}" y="{search_top + search_height/2 + 4:.2f}" text-anchor="middle">{search:.2f}</text>'
            )
        else:
            svg.append(
                f'<text class="axis" x="{x + bar_width + 8:.2f}" y="{search_top + 2:.2f}">Search {search:.2f}</text>'
            )
        svg.append(
            f'<text class="segment" x="{centers[mode]:.2f}" y="{total_top + wait_height/2 + 4:.2f}" text-anchor="middle">{wait:.2f}</text>'
        )
        svg.append(
            f'<text class="total" x="{centers[mode]:.2f}" y="{total_top-10:.2f}" text-anchor="middle">Total {total:.2f} ms</text>'
        )
        svg.append(
            f'<text class="axis" x="{centers[mode]:.2f}" y="{baseline_y+30:.2f}" text-anchor="middle">{escape(mode_labels[mode])}</text>'
        )

    legend_y = height - 34
    for index, (key, label) in enumerate(
        (("search", "HNSW search + inference"), ("wait", "Raw-data ready wait"))
    ):
        legend_x = 145 + index * 300
        svg.append(
            f'<rect x="{legend_x}" y="{legend_y-14}" width="20" height="20" rx="2" fill="{colors[key]}"/>'
        )
        svg.append(
            f'<text class="legend" x="{legend_x+29}" y="{legend_y+1}">{escape(label)}</text>'
        )

    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
