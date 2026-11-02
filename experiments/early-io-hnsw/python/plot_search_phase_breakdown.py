"""Aggregate per-query search phases and plot a stacked model comparison PNG."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ("baseline", "lstm_2layer", "lstm_1layer", "gru")
LABELS = ("Baseline", "2-layer LSTM", "1-layer LSTM", "GRU")
PHASES = ("graph_ms", "feature_ms", "snapshot_and_other_ms", "io_wait_ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[10])
    parser.add_argument("--split-by-k", action="store_true",
                        help="read each k from result-root/search_breakdown_k<K>/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    records = []
    for k in args.k:
        source_root = (args.result_root / f"search_breakdown_k{k}"
                       if args.split_by_k else args.result_root)
        for mode in MODES:
            path = source_root / f"k{k}_{mode}" / "search_breakdown.csv"
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            if not rows or len({row["query_id"] for row in rows}) != len(rows):
                raise ValueError(f"Empty or duplicated query records: {path}")
            keys = (*PHASES, "enqueue_ms", "search_ms", "e2e_ms")
            for row in rows:
                values = {key: float(row[key]) for key in keys}
                if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                    raise ValueError(f"Invalid latency in {path}")
                if not math.isclose(sum(values[p] for p in PHASES[:3]),
                                    values["search_ms"], abs_tol=5e-9):
                    raise ValueError(f"Search phases do not sum to search: {path}")
                if not math.isclose(sum(values[p] for p in PHASES),
                                    values["e2e_ms"], abs_tol=5e-9):
                    raise ValueError(f"Phases do not sum to E2E: {path}")
            records.append({"k": k, "mode": mode, "queries": len(rows),
                            **{key: sum(float(row[key]) for row in rows) / len(rows)
                               for key in keys}})
    if len({record["queries"] for record in records}) != 1:
        raise ValueError("Query counts differ across conditions")
    summary = args.result_root / "search_phase_summary_ms.tsv"
    with summary.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    (args.result_root / "search_phase_summary_ms.json").write_text(
        json.dumps({"units": "ms", "results": records}, indent=2) + "\n")

    positions = [5 * group + rank for group in range(len(args.k)) for rank in range(4)]
    fig, ax = plt.subplots(figsize=(max(10, 6 * len(args.k)), 6))
    bottom = [0.] * len(records)
    colors = ("#2563eb", "#22c55e", "#a855f7", "#f97316")
    names = ("Graph traversal", "Feature calculation / collection",
             "Snapshot delivery / other", "Final I/O wait")
    largest = max(row["e2e_ms"] for row in records)
    for phase, color, label in zip(PHASES, colors, names):
        values = [row[phase] for row in records]
        ax.bar(positions, values, bottom=bottom, width=.72, color=color, label=label)
        for x, height, base in zip(positions, values, bottom):
            if height >= largest * .035:
                ax.text(x, base + height / 2, f"{height:.3f}", ha="center",
                        va="center", fontsize=9, color="white")
        bottom = [a + b for a, b in zip(bottom, values)]
    for x, row in zip(positions, records):
        ax.annotate(f"Total: {row['e2e_ms']:.3f} ms", (x, row["e2e_ms"]),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(positions, list(LABELS) * len(args.k))
    for group, k in enumerate(args.k):
        ax.text(5 * group + 1.5, -.10, f"k = {k}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean latency per query (ms)")
    ax.set_ylim(0, largest * 1.20)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.24), ncol=2, frameon=False)
    fig.subplots_adjust(left=.08, right=.99, bottom=.18, top=.76)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(summary)
    print(args.output)


if __name__ == "__main__":
    main()
