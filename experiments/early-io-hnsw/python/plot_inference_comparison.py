#!/usr/bin/env python3
"""Create an SVG grouped bar chart from prefetch on/off result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def latency_ms(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return float(result["elapsed_seconds"]) * 1000.0 / int(result["test_query_count"])


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    output = Path(args.output).resolve()
    topks = (10, 20, 50)
    values = {
        topk: {
            mode: latency_ms(
                model_root
                / f"k{topk}"
                / f"inference_metrics_prefetch_{mode}.json"
            )
            for mode in ("off", "on")
        }
        for topk in topks
    }

    width, height = 900, 560
    left, right, top, bottom = 100, 40, 80, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(value for group in values.values() for value in group.values())
    tick_step = 50.0
    y_max = max(tick_step, math.ceil(maximum / tick_step) * tick_step)

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / y_max)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}.axis{font-size:13px}.value{font-size:12px;font-weight:600}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2}" y="32" text-anchor="middle">End-to-End Latency per Query</text>',
        f'<text class="subtitle" x="{width / 2}" y="54" text-anchor="middle">2,000 test queries · threshold 0.5 · lower is better</text>',
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

    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#374151" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#374151" stroke-width="1.5"/>',
        f'<text class="axis" x="22" y="{top + plot_height/2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height/2})">Latency (ms/query)</text>',
    ])

    group_width = plot_width / len(topks)
    bar_width = 78
    bar_gap = 18
    colors = {"off": "#6b7280", "on": "#2563eb"}
    labels = {"off": "Baseline", "on": "Prefetch"}
    for group_index, topk in enumerate(topks):
        center = left + group_width * (group_index + 0.5)
        for mode_index, mode in enumerate(("off", "on")):
            value = values[topk][mode]
            x = center + (mode_index - 0.5) * (bar_width + bar_gap) - bar_width / 2
            bar_top = y(value)
            bar_height = top + plot_height - bar_top
            svg.append(
                f'<rect x="{x:.2f}" y="{bar_top:.2f}" width="{bar_width}" height="{bar_height:.2f}" rx="3" fill="{colors[mode]}"/>'
            )
            svg.append(
                f'<text class="value" x="{x+bar_width/2:.2f}" y="{bar_top-8:.2f}" text-anchor="middle">{value:.2f}</text>'
            )
        svg.append(
            f'<text class="axis" x="{center:.2f}" y="{top+plot_height+28}" text-anchor="middle">k={topk}</text>'
        )

    legend_y = height - 28
    for index, mode in enumerate(("off", "on")):
        legend_x = width / 2 - 130 + index * 150
        svg.append(
            f'<rect x="{legend_x:.2f}" y="{legend_y-13}" width="18" height="18" rx="2" fill="{colors[mode]}"/>'
        )
        svg.append(
            f'<text class="legend" x="{legend_x+27:.2f}" y="{legend_y+1}">{labels[mode]}</text>'
        )

    svg.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
