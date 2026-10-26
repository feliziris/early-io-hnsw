#!/usr/bin/env python3
"""Plot critical-path phases and overlapping async inference for one query."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_gpu_pipeline(data, output):
    """CPU lane timings only: Run wall time is not pure GPU compute time."""
    events = data["inference_events"]
    phases = data["phases_ms"]
    if not events:
        raise ValueError("No inference events")
    sums = {key: sum(e[key] for e in events) for key in (
        "input_tensor_ms", "session_run_ms", "transfer_enqueue_ms",
        "state_update_ms", "collect_wait_ms", "probability_copy_ms",
    )}
    submit_total = sum(e["gpu_enqueued_ms"] - e["started_ms"] for e in events)
    collect_total = sum(e["predicted_ms"] - e["collect_started_ms"] for e in events)
    threshold_total = sum(e["submitted_ms"] - e["predicted_ms"] for e in events)
    worker_span = events[-1]["submitted_ms"] - events[0]["started_ms"]
    idle_other = worker_span - sum(sums.values()) - threshold_total
    if idle_other < -1e-6:
        raise ValueError("Worker timing components overlap")
    waits = [e["started_ms"] - e["enqueued_ms"] for e in events]
    summary = {
        "query_row": data["query_row"], "k": data["k"],
        "timesteps": len(events), "provider": data["provider"],
        "end_to_end_ms": data["end_to_end_ms"],
        "search_thread_phases_ms": phases,
        "worker_span_ms": worker_span,
        "worker_first_start_ms": events[0]["started_ms"],
        "worker_last_io_submit_ms": events[-1]["submitted_ms"],
        "worker_components_ms": {
            **sums, "threshold_and_io_submit": threshold_total,
            "other_including_idle_and_polling": max(0, idle_other),
        },
        "gpu_submit_host_total_ms": submit_total,
        "gpu_collect_host_total_ms": collect_total,
        "fifo_wait_mean_ms": sum(waits) / len(waits),
        "fifo_wait_max_ms": max(waits),
        "peak_snapshot_queue": data["peak_snapshot_queue"],
        "gpu_peak_inflight": data["gpu_peak_inflight"],
        "note": "CPU thread wall-clock profiling; Session::Run includes host launch overhead and any internal waits, not pure GPU kernel duration. Overlapping thread totals must not be added. Instrumented single-query diagnostic, not a throughput benchmark.",
    }
    fig, axes = plt.subplots(4, 1, figsize=(15, 12),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.4]})
    names = ["Upper levels", "Query state / GPU buffer setup", "Level-0 traversal",
             "Remaining inference + writeback", "Final extraction",
             "Final request lookup / submit", "Final payload wait"]
    colors = ["#8b5cf6", "#64748b", "#2563eb", "#60a5fa", "#0d9488", "#fbbf24", "#f97316"]
    left = 0
    for (_, duration), label, color in zip(phases.items(), names, colors):
        axes[0].barh(0, duration, left=left, height=.5, color=color,
                     label=f"{label}: {duration:.3f} ms")
        if duration > data["end_to_end_ms"] * .1:
            axes[0].text(left + duration / 2, 0, f"{duration:.2f}", ha="center", va="center")
        left += duration
    axes[0].set_yticks([0], ["Search thread"])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 2.2), ncol=3, frameon=False, fontsize=9)
    for start, end, color, label in [
        ("started_ms", "gpu_enqueued_ms", "#16a34a", "CPU preparation + GPU submission"),
        ("collect_started_ms", "predicted_ms", "#7c3aed", "GPU result collection / wait"),
        ("predicted_ms", "submitted_ms", "#f97316", "Threshold + I/O submission"),
    ]:
        axes[1].broken_barh([(e[start], e[end] - e[start]) for e in events],
                            (-.25, .5), facecolors=color, label=label)
    axes[1].set_yticks([0], ["Inference thread"])
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, 1.7), ncol=3, frameon=False, fontsize=9)
    for ax in axes[:2]:
        ax.set_xlim(0, data["end_to_end_ms"] * 1.01)
        ax.set_xlabel("Elapsed time from query start (ms)")
        ax.axvline(data["boundaries_ms"]["graph_end"], color="#dc2626", linestyle="--", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)

    labels = ["Input copy + I/O binding", "Session::Run host call", "Transfer / event enqueue",
              "State pointer update", "Completion check / wait", "Probability copy",
              "Threshold + I/O submit", "Other / idle / polling"]
    values = list(sums.values()) + [threshold_total, max(0, idle_other)]
    palette = ["#0d9488", "#16a34a", "#38bdf8", "#6366f1", "#7c3aed", "#eab308", "#f97316", "#cbd5e1"]
    left = 0
    for label, value, color in zip(labels, values, palette):
        axes[2].barh(0, value, left=left, height=.5, color=color,
                     label=f"{label}: {value:.3f} ms")
        if value > worker_span * .12:
            axes[2].text(left + value / 2, 0, f"{value:.2f}", ha="center", va="center")
        left += value
    axes[2].set_yticks([0], ["Inference totals"])
    axes[2].set_xlabel("Accumulated CPU-side wall time (ms), not GPU kernel time")
    axes[2].set_xlim(left=0)
    axes[2].legend(loc="upper left", bbox_to_anchor=(0, 2.2), ncol=3, frameon=False, fontsize=9)
    axes[3].plot([e["timestep"] for e in events], waits, color="#7c3aed")
    axes[3].set_xlabel("Snapshot timestep (zero-based)")
    axes[3].set_ylabel("FIFO wait before GPU submission (ms)")
    axes[3].set_ylim(bottom=0)
    axes[3].grid(alpha=.25)
    fig.text(.14, .025,
             f"k={data['k']} | query row {data['query_row']} | {len(events)} steps | "
             f"E2E {data['end_to_end_ms']:.3f} ms | FIFO peak {data['peak_snapshot_queue']} | "
             f"GPU pending peak {data['gpu_peak_inflight']}\n"
             f"Within level 0: features {data['nested_work_ms']['feature']:.3f} ms; "
             f"snapshot enqueue {data['nested_work_ms']['snapshot_enqueue']:.3f} ms. "
             "Dashed red: traversal complete.\n"
             "Search and inference lanes overlap: do not add their durations. GPU/IO execute outside these CPU lanes.",
             fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.13, hspace=1.4, left=.15, right=.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    summary_path = output.parent.parent / (output.stem + ".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)
    print(summary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.profile.read_text())
    phases = data["phases_ms"]
    if abs(sum(phases.values()) - data["end_to_end_ms"]) > 1e-6:
        raise ValueError("Critical-path phases do not sum to end-to-end time")
    for name, duration in phases.items():
        if duration < 0:
            raise ValueError(f"Negative phase: {name}")
    if data.get("gpu_inflight_limit", 0):
        plot_gpu_pipeline(data, args.output)
        return
    labels = [
        "Upper levels", "Query state / FIFO setup", "Level-0 traversal",
        "Remaining inference + trace writeback", "Final top-k extraction",
        "Final request lookup / submission", "Final payload wait",
    ]
    colors = ["#8b5cf6", "#64748b", "#2563eb", "#60a5fa", "#0d9488", "#fbbf24", "#f97316"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1, 1.7]})
    left = 0
    for (name, duration), label, color in zip(phases.items(), labels, colors):
        axes[0].barh(0, duration, left=left, color=color, height=0.5,
                     label=f"{label}: {duration:.3f} ms")
        if duration > data["end_to_end_ms"] * 0.08:
            axes[0].text(left + duration / 2, 0, f"{duration:.3f} ms", ha="center", va="center")
        left += duration
    axes[0].set_yticks([0], ["Search thread"])
    axes[0].set_xlabel("Elapsed time from query start (ms)")
    axes[0].set_xlim(0, data["end_to_end_ms"] * 1.02)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 2.0), ncol=3, frameon=False, fontsize=9)

    events = data["inference_events"]
    predict = [(e["started_ms"], e["predicted_ms"] - e["started_ms"]) for e in events]
    submit = [(e["predicted_ms"], e["submitted_ms"] - e["predicted_ms"]) for e in events]
    axes[1].broken_barh(predict, (-0.25, 0.5), facecolors="#16a34a", label="predict()")
    axes[1].broken_barh(submit, (-0.25, 0.5), facecolors="#f97316", label="Threshold + I/O submission")
    axes[1].set_yticks([0], ["Inference worker"])
    axes[1].set_xlabel("Elapsed time from query start (ms)")
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.3), ncol=2)
    for ax in axes[:2]:
        ax.axvline(data["boundaries_ms"]["graph_end"], color="#dc2626", linestyle="--", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    waits = [e["started_ms"] - e["enqueued_ms"] for e in events]
    axes[2].plot([e["timestep"] for e in events], waits, color="#7c3aed")
    axes[2].set_xlabel("Snapshot timestep (zero-based)")
    axes[2].set_ylabel("FIFO wait before predict() (ms)")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.25)
    nested = data["nested_work_ms"]
    fig.text(0.13, 0.015,
             f"Query row {data['query_row']} | k={data['k']} | {data['timesteps']} timesteps | "
             f"E2E {data['end_to_end_ms']:.3f} ms | peak FIFO {data['peak_snapshot_queue']}\n"
             f"Within traversal: features {nested['feature']:.3f} ms, enqueue {nested['snapshot_enqueue']:.3f} ms. "
             f"Overlapping predict total {nested['predict_total']:.3f} ms.\n"
             "Dashed red line: graph traversal ended. I/O reads overlap these lanes; read durations are not stacked.",
             fontsize=9)
    fig.subplots_adjust(top=0.82, bottom=0.16, hspace=1.0, left=0.14, right=0.97)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
