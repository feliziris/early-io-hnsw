#!/usr/bin/env python3
"""Summarize k=10/20/50 prefetch on/off benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    output = Path(args.output).resolve()
    rows = []
    paired = []
    for topk in (10, 20, 50):
        results = {}
        for mode in ("off", "on"):
            path = model_root / f"k{topk}" / f"inference_metrics_prefetch_{mode}.json"
            result = load(path)
            results[mode] = result
            wait = result["raw_io"]["query_data_ready_wait_latency"]
            rows.append({
                "topk": topk,
                "mode": mode,
                "elapsed": result["elapsed_seconds"],
                "wait_mean": wait["mean_us"],
                "wait_p95": wait["p95_us"],
                "wait_p99": wait["p99_us"],
                "async_reads": result["raw_io"]["async_prefetch_read_count"],
                "demand_reads": result["raw_io"]["synchronous_demand_read_count"],
                "total_bytes": result["raw_io"]["total_bytes_read"],
                "wasted": result["metrics"]["wasted_io_ratio"],
                "tp_delay": result["metrics"]["tp_decision_delay"],
            })
        off = results["off"]
        on = results["on"]
        off_wait = off["raw_io"]["query_data_ready_wait_latency"]["mean_us"]
        on_wait = on["raw_io"]["query_data_ready_wait_latency"]["mean_us"]
        paired.append({
            "topk": topk,
            "elapsed_change": (on["elapsed_seconds"] / off["elapsed_seconds"] - 1.0) * 100.0,
            "wait_reduction": (1.0 - on_wait / off_wait) * 100.0 if off_wait else 0.0,
        })

    lines = [
        "# Offline prefetch benchmark",
        "",
        "![End-to-end latency per query](figure/end_to_end_latency_per_query.svg)",
        "",
        "The chart uses `elapsed_seconds / 2,000` and starts the y-axis at zero.",
        "",
        "- Test queries: 2,000 per run (embedded HDF5 test split)",
        "- Threshold: 0.5",
        "- Raw payload: 512 KiB per node",
        "- Async I/O workers: 4",
        "- Cache control: `POSIX_FADV_DONTNEED` advised before every mmap",
        "- Scope: offline inference and final top-k data-ready wait; live Faiss HNSW search is not included",
        "",
        "## Measurements",
        "",
        "| k | Prefetch | Total elapsed (s) | Query wait mean (us) | p95 (us) | p99 (us) | Async reads | Demand reads | Total read (GiB) | Wasted I/O ratio | TP delay |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['topk']} | {row['mode']} | {row['elapsed']:.3f} | "
            f"{row['wait_mean']:.3f} | {row['wait_p95']:.3f} | "
            f"{row['wait_p99']:.3f} | {row['async_reads']} | "
            f"{row['demand_reads']} | {row['total_bytes'] / 2**30:.3f} | "
            f"{row['wasted']:.6f} | {row['tp_delay']:.6f} |"
        )
    lines.extend([
        "",
        "## On vs. off",
        "",
        "Positive query-wait reduction means prefetch reduced final data waiting. "
        "Positive elapsed change means the prefetch-on run took longer overall.",
        "",
        "| k | Total elapsed change (%) | Query wait reduction (%) |",
        "|---:|---:|---:|",
    ])
    for item in paired:
        lines.append(
            f"| {item['topk']} | {item['elapsed_change']:.3f} | "
            f"{item['wait_reduction']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation caveat",
        "",
        "`POSIX_FADV_DONTNEED` is a per-file eviction hint, not a kernel guarantee. "
        "The timings can still contain page-cache hits. For stronger evidence, repeat "
        "the complete experiment several times and report variability.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
