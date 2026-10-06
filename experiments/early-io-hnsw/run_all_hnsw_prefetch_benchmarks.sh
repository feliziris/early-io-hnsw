#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${script_dir}/../../.." && pwd)"
runner="${script_dir}/run_hnsw_prefetch_benchmark.sh"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_fair_workers4}"
result_root="${RESULT_ROOT:-${workspace_root}/results/hnsw_live_prefetch/${run_id}}"
repetitions="${REPETITIONS:-1}"
batch_size="${BATCH_SIZE:-1}"
threshold="${THRESHOLD:-0.5}"

mkdir -p "${result_root}"
for topk in 10 20 50; do
    echo "start topk=${topk}"
    RESULT_DIR="${result_root}/k${topk}" \
        "${runner}" "${repetitions}" "${topk}" "${batch_size}" "${threshold}"
    echo "done topk=${topk}"
done

echo "all benchmarks complete"
echo "result_root=${result_root}"
