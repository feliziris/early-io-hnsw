import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_ms(path):
    fields = dict(line.split("=", 1) for line in
                  path.read_text().splitlines() if "=" in line)
    keys = ["raw_search_mean_us", "raw_data_ready_wait_mean_us",
            "raw_end_to_end_mean_us"]
    values = [float(fields[key]) / 1000 for key in keys]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError(f"Invalid latency in {path}")
    if not math.isclose(values[0] + values[1], values[2],
                        rel_tol=1e-4, abs_tol=1e-5):
        raise ValueError(f"Search + I/O wait does not match E2E in {path}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--result-root", type=Path, default=Path("workspace/results"))
    baseline_source = parser.add_mutually_exclusive_group()
    baseline_source.add_argument("--baseline-log", type=Path, nargs="+")
    baseline_source.add_argument("--baseline-dir", type=Path, nargs="+",
                                 help="Directory containing run.log; a bare name is relative to --result-root")
    prefetch_source = parser.add_mutually_exclusive_group()
    prefetch_source.add_argument("--prefetch-log", type=Path, nargs="+")
    prefetch_source.add_argument("--prefetch-dir", type=Path, nargs="+",
                                 help="Directory containing run.log; a bare name is relative to --result-root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(k <= 0 for k in args.k) or len(set(args.k)) != len(args.k):
        parser.error("--k values must be positive and unique")
    def directory_log(directory):
        if not directory.is_absolute() and len(directory.parts) == 1:
            directory = args.result_root / directory
        return directory / "run.log"

    def logs_for(mode):
        logs = getattr(args, f"{mode}_log")
        directories = getattr(args, f"{mode}_dir")
        paths = logs or [directory_log(directory) for directory in directories or [
            args.result_root / f"k{k}_{mode}" for k in args.k
        ]]
        if len(paths) != len(args.k):
            parser.error(f"Provide one {mode} log/directory per --k, in the same order")
        return paths

    baseline = logs_for("baseline")
    prefetch = logs_for("prefetch")
    k_label = "_".join(f"k{k}" for k in args.k)
    output = args.output or args.result_root / f"figure/{k_label}_latency_breakdown.png"
    if output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    try:
        data = [read_ms(path) for pair in zip(baseline, prefetch) for path in pair]
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    search, wait, total = zip(*data)
    positions = [3 * group + offset for group in range(len(args.k)) for offset in (0, 1)]
    fig, ax = plt.subplots(figsize=(max(7, 4 * len(args.k)), 5.5))
    ax.bar(positions, search, width=.65, color="#2563eb", label="Search / inference phase")
    ax.bar(positions, wait, bottom=search, width=.65, color="#f97316", label="Final I/O wait")
    for i, (s, w, t) in zip(positions, data):
        if s > max(total) * .08:
            ax.text(i, s / 2, f"{s:.3f} ms", ha="center", va="center", color="white")
        ax.annotate(f"Total: {t:.3f} ms\nI/O wait: {w:.3f} ms", (i, s + w),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(positions, ["Baseline", "Prefetch"] * len(args.k))
    for group, k in enumerate(args.k):
        ax.text(3 * group + .5, -.12, f"k = {k}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontweight="bold")
    ax.set_ylabel("Mean latency per query (ms)")
    ax.set_ylim(0, max(max(total) * 1.35, .001))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    for group, k in enumerate(args.k):
        print(f"k={k} baseline_log={baseline[group]} prefetch_log={prefetch[group]}")
        for name, (s, w, t) in zip(["Baseline", "Prefetch"], data[2 * group:2 * group + 2]):
            print(f"k={k} {name}: search={s:.6f}, I/O wait={w:.6f}, E2E={t:.6f} ms")
    print(output)


if __name__ == "__main__":
    main()
