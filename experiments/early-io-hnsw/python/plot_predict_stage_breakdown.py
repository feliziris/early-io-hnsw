#!/usr/bin/env python3
"""Plot per-snapshot CPU inference stages; write PNG and numeric JSON only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    data = json.loads(args.profile.read_text())
    events = data["inference_events"]
    if not events:
        raise ValueError("No inference events")
    # Run includes the CPU LSTM and every other ONNX graph operator. Its
    # dispatch and model computation cannot be separated by outer wall clocks.
    names = ["Input tensor preparation", "Session::Run() incl. LSTM",
             "Output probability copy", "Hidden / cell state update",
             "Other predict() work", "Threshold + I/O submission"]
    colors = ["#8b5cf6", "#16a34a", "#0ea5e9", "#2563eb", "#64748b", "#f97316"]
    keys = ["input_tensor_ms", "session_run_ms", "probability_copy_ms", "state_update_ms"]
    values = [[e[key] for e in events] for key in keys]
    others = []
    for e in events:
        residual = e["predicted_ms"] - e["started_ms"] - sum(e[key] for key in keys)
        if residual < -1e-6:
            raise ValueError("Stage times exceed predict() wall time")
        others.append(max(0, residual))
    values.extend([others, [e["submitted_ms"] - e["predicted_ms"] for e in events]])
    matrix = np.asarray(values, dtype=float)
    if np.any(matrix < 0):
        raise ValueError("Negative duration")
    snapshot_total = np.asarray([e["submitted_ms"] - e["started_ms"] for e in events])
    if not np.allclose(matrix.sum(axis=0), snapshot_total, atol=1e-6, rtol=0):
        raise ValueError("Stages do not add up to observed snapshot processing")
    means = matrix.mean(axis=1)
    totals = matrix.sum(axis=1)
    summary = {
        "profile": str(args.profile), "k": data["k"],
        "query_row": data["query_row"], "snapshots": len(events),
        "unit": "ms", "scope": "after snapshot pop, before predict through I/O submit",
        "fifo_wait_included": False, "payload_read_included": False,
        "run_includes_lstm_and_all_onnx_operators": True,
        "other_predict_work": "output metadata checks, output tensor cleanup, return and timing overhead",
        "mean_snapshot_processing_ms": float(snapshot_total.mean()),
        "total_snapshot_processing_ms": float(snapshot_total.sum()),
        "stages": {
            name: {"mean_ms": float(mean), "total_ms": float(total),
                   "fraction": float(total / snapshot_total.sum())}
            for name, mean, total in zip(names, means, totals)
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 1.2]})
    positions = np.arange(len(names))
    axes[0].barh(positions, means, color=colors, height=0.65)
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    for i, mean in enumerate(means):
        axes[0].annotate(f"{mean:.6f} ms", (mean, i), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=10)
    axes[0].set_xlim(0, max(means) * 1.25)
    axes[0].set_xlabel("Mean stage latency per snapshot (ms)")
    x = np.asarray([e["timestep"] for e in events])
    axes[1].stackplot(x, matrix, labels=names, colors=colors)
    axes[1].set_xlabel("Snapshot timestep (zero-based)")
    axes[1].set_ylabel("Snapshot processing latency (ms)")
    axes[1].set_xlim(x[0], x[-1])
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right", frameon=True, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.text(0.23, 0.025,
             f"CPU | k={data['k']} | query row {data['query_row']} | {len(events)} snapshots | "
             f"mean processing {snapshot_total.mean():.6f} ms/snapshot\n"
             "FIFO wait and payload reading excluded. Session::Run() includes LSTM computation.\n"
             "Timing enabled; individual calls and per-stage clock readings add measurement overhead.",
             fontsize=9)
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.14, top=0.97, hspace=0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
