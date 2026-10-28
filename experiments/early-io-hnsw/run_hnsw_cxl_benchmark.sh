#!/usr/bin/env bash
# Usage: ./run_hnsw_cxl_benchmark.sh K CPU_NODE
# CPU_NODE is a NUMA node (--cpunodebind), not a logical CPU ID.
set -euo pipefail

if [[ $# != 2 || ! "$1" =~ ^[1-9][0-9]*$ || ! "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 K CPU_NODE" >&2
    exit 1
fi

topk="$1"
cpu="$2"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd -- "${repo_root}"

generator="${GENERATOR_PATH:-${repo_root}/build-onnx-cpu/demos/generate_sift_query_labels}"
index="${INDEX_PATH:-${repo_root}/data/index/bigann_base10M_hnsw_M32_efC128.faiss}"
queries="${QUERY_PATH:-${repo_root}/output/sift10M/bigann_query_test1000_seed42.fvecs}"
sidecar="${AVG_NEIGHBOR_DISTANCE_PATH:-${repo_root}/data/index/bigann_base10M_hnsw_M32_efC128.level0_avg_neighbor_distance.bin}"
model="${ONNX_MODEL_PATH:-${repo_root}/model/sift10M/lstm_io/ef200/k${topk}/model.onnx}"
raw_data="${RAW_DATA_PATH:-${repo_root}/workspace/raw_data/payload4k_10m.bin}"
result_root="${RESULT_ROOT:-${repo_root}/workspace/results}"
memory_node="${MEMORY_NODE:-2}"

command -v numactl >/dev/null || { echo "numactl is required" >&2; exit 1; }
[[ -x "${generator}" ]] || { echo "Executable not found: ${generator}" >&2; exit 1; }
for input in "${index}" "${queries}" "${sidecar}" "${model}" "${raw_data}"; do
    [[ -r "${input}" ]] || { echo "Input not readable: ${input}" >&2; exit 1; }
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

for mode in baseline prefetch; do
    result_dir="${result_root}/k${topk}_${mode}_cxl_cpu${cpu}"
    mkdir -p -- "${result_dir}"
    command=(numactl --cpunodebind "${cpu}" --membind "${memory_node}"
        "${generator}" "${index}" "${queries}"
        "${result_dir}/trace.bin" "${result_dir}/trace_index.tsv"
        "${topk}" "${EF_SEARCH:-200}" 1
        --avg-neighbor-distance "${sidecar}"
        --raw-data "${raw_data}" --payload-size "${PAYLOAD_SIZE:-4096}"
        --io-workers "${IO_WORKERS:-4}" --io-queue-size "${IO_QUEUE_SIZE:-4096}"
        --evict-file-cache --warmup-search)
    if [[ "${mode}" == prefetch ]]; then
        command+=(--prefetch on --onnx-model "${model}"
            --onnx-provider cpu --onnx-inference async --threshold "${THRESHOLD:-0.5}")
    else
        command+=(--prefetch off)
    fi
    # Record the exact command and thread settings for reproducibility.
    {
        printf 'OMP_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q ' "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "${result_dir}/command.sh"
    echo "Running ${mode}: k=${topk}, CPU node=${cpu}, memory node=${memory_node}"
    "${command[@]}" 2>&1 | tee "${result_dir}/run.log"
done
