#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
inference="${project_root}/python/inference.py"
model_root="${project_root}/model/sift/lstm_io"

for topk in 10 20 50; do
    for mode in off on; do
        output="${model_root}/k${topk}/inference_metrics_prefetch_${mode}.json"
        if [[ -s "${output}" ]]; then
            echo "skip existing: k=${topk} prefetch=${mode} output=${output}"
            continue
        fi
        echo "start: k=${topk} prefetch=${mode}"
        "${python_bin}" "${inference}" \
            --topk "${topk}" \
            --threshold 0.5 \
            --prefetch "${mode}" \
            --io-workers 4 \
            --evict-file-cache \
            --output "${output}"
        echo "done: k=${topk} prefetch=${mode}"
    done
done

"${python_bin}" "${project_root}/python/summarize_inference.py" \
    --model-root "${model_root}" \
    --output "${project_root}/results/inference_comparison.md"

"${python_bin}" "${project_root}/python/plot_inference_comparison.py" \
    --model-root "${model_root}" \
    --output "${project_root}/results/figure/end_to_end_latency_per_query.svg"

echo "benchmark complete"
