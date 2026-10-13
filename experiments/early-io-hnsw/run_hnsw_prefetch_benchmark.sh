#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
faiss_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${faiss_root}/.." && pwd)"
project_root="$(cd "${workspace_root}/.." && pwd)"

repetitions="${1:-1}"
topk="${2:-10}"
batch_size="${3:-1}"
threshold="${4:-0.5}"
onnx_provider="${ONNX_PROVIDER:-cpu}"
onnx_device_id="${ONNX_DEVICE_ID:-0}"

ef_search="${EF_SEARCH:-200}"
io_workers="${IO_WORKERS:-4}"
io_queue_size="${IO_QUEUE_SIZE:-4096}"
payload_size="${PAYLOAD_SIZE:-524288}"
build_dir="${BUILD_DIR:-${project_root}/build-onnx}"
generator="${build_dir}/demos/generate_sift_query_labels"
index="${INDEX_PATH:-${project_root}/output/sift100k_hnsw_m32_efc200_efs200.index}"
queries="${QUERY_PATH:-${project_root}/output/sift_query_test_seed42.fvecs}"
raw_data="${RAW_DATA_PATH:-${workspace_root}/raw_data/payload512k.bin}"
onnx_model="${ONNX_MODEL_PATH:-${workspace_root}/model/sift/lstm_io/k${topk}/model.onnx}"
avg_neighbor_distance="${AVG_NEIGHBOR_DISTANCE_PATH:-}"
if [[ -z "${avg_neighbor_distance}" && -f "${index%.*}.level0_avg_neighbor_distance.bin" ]]; then
    avg_neighbor_distance="${index%.*}.level0_avg_neighbor_distance.bin"
fi

run_id="$(date +%Y%m%d_%H%M%S)_$$"
result_dir="${RESULT_DIR:-${workspace_root}/results/hnsw_live_prefetch/${run_id}}"
summary="${result_dir}/summary.tsv"
temporary_dir="$(mktemp -d /tmp/hnsw-prefetch-benchmark.XXXXXX)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "required file not found: $1" >&2
        exit 1
    fi
}

require_positive_integer() {
    if [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
        echo "$2 must be a positive integer: $1" >&2
        exit 1
    fi
}

metric() {
    local name="$1"
    local log="$2"
    awk -F= -v key="${name}" '$1 == key { value = $2 } END { print value }' "${log}"
}

require_positive_integer "${repetitions}" "repetitions"
require_positive_integer "${topk}" "topk"
require_positive_integer "${batch_size}" "batch-size"
require_file "${generator}"
require_file "${index}"
require_file "${queries}"
require_file "${raw_data}"
require_file "${onnx_model}"
if [[ -n "${avg_neighbor_distance}" ]]; then
    require_file "${avg_neighbor_distance}"
fi

mkdir -p "${result_dir}"
printf 'round\tmode\tsearch_mean_us\tdata_ready_wait_mean_us\tend_to_end_mean_us\ttp\tfp\tfn\twasted_io_ratio\ttotal_bytes_read\tlog\n' >"${summary}"

run_one() {
    local round="$1"
    local mode="$2"
    local log="${result_dir}/round_${round}_prefetch_${mode}.log"
    local trace="${temporary_dir}/trace.bin"
    local trace_index="${temporary_dir}/trace_index.tsv"
    local command=(
        "${generator}"
        "${index}"
        "${queries}"
        "${trace}"
        "${trace_index}"
        "${topk}"
        "${ef_search}"
        "${batch_size}"
    )
    if [[ -n "${avg_neighbor_distance}" ]]; then
        command+=(--avg-neighbor-distance "${avg_neighbor_distance}")
    fi

    if [[ "${mode}" == "on" ]]; then
        command+=(
            --onnx-model "${onnx_model}"
            --onnx-provider "${onnx_provider}"
            --onnx-device-id "${onnx_device_id}"
            --threshold "${threshold}"
        )
    fi
    command+=(
        --prefetch "${mode}"
        --raw-data "${raw_data}"
        --payload-size "${payload_size}"
        --io-workers "${io_workers}"
        --io-queue-size "${io_queue_size}"
        --evict-file-cache
    )

    echo "start round=${round} prefetch=${mode}"
    OMP_NUM_THREADS=1 "${command[@]}" 2>&1 | tee "${log}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${round}" \
        "${mode}" \
        "$(metric raw_search_mean_us "${log}")" \
        "$(metric raw_data_ready_wait_mean_us "${log}")" \
        "$(metric raw_end_to_end_mean_us "${log}")" \
        "$(metric raw_prefetch_tp "${log}")" \
        "$(metric raw_prefetch_fp "${log}")" \
        "$(metric raw_prefetch_fn "${log}")" \
        "$(metric raw_wasted_io_ratio "${log}")" \
        "$(metric raw_total_bytes_read "${log}")" \
        "${log}" >>"${summary}"
    echo "done round=${round} prefetch=${mode}"
}

cat >"${result_dir}/configuration.txt" <<EOF
repetitions=${repetitions}
topk=${topk}
ef_search=${ef_search}
batch_size=${batch_size}
threshold=${threshold}
io_workers=${io_workers}
io_queue_size=${io_queue_size}
payload_size=${payload_size}
generator=${generator}
index=${index}
queries=${queries}
raw_data=${raw_data}
onnx_model=${onnx_model}
onnx_provider=${onnx_provider}
onnx_device_id=${onnx_device_id}
avg_neighbor_distance=${avg_neighbor_distance:-computed_at_startup}
omp_num_threads=1
evict_file_cache=true
EOF

for ((round = 1; round <= repetitions; ++round)); do
    run_one "${round}" off
    run_one "${round}" on
done

echo "benchmark complete"
echo "summary=${summary}"
