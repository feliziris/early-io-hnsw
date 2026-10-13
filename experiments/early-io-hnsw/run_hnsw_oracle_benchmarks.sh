#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
faiss_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${faiss_root}/.." && pwd)"
project_root="$(cd "${workspace_root}/.." && pwd)"

ef_search="${EF_SEARCH:-200}"
batch_size="${BATCH_SIZE:-1}"
io_workers="${IO_WORKERS:-4}"
io_queue_size="${IO_QUEUE_SIZE:-4096}"
payload_size="${PAYLOAD_SIZE:-524288}"
build_dir="${BUILD_DIR:-${project_root}/build-onnx}"
generator="${build_dir}/demos/generate_sift_query_labels"
index="${INDEX_PATH:-${project_root}/output/sift100k_hnsw_m32_efc200_efs200.index}"
queries="${QUERY_PATH:-${project_root}/output/sift_query_test_seed42.fvecs}"
raw_data="${RAW_DATA_PATH:-${workspace_root}/raw_data/payload512k.bin}"
avg_neighbor_distance="${AVG_NEIGHBOR_DISTANCE_PATH:-}"
if [[ -z "${avg_neighbor_distance}" && -f "${index%.*}.level0_avg_neighbor_distance.bin" ]]; then
    avg_neighbor_distance="${index%.*}.level0_avg_neighbor_distance.bin"
fi
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_oracle_workers4}"
result_root="${RESULT_ROOT:-${workspace_root}/results/hnsw_live_prefetch/${run_id}}"
summary="${result_root}/oracle_summary.tsv"
temporary_dir="$(mktemp -d /tmp/hnsw-oracle-benchmark.XXXXXX)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

if [[ -n "${avg_neighbor_distance}" && ! -f "${avg_neighbor_distance}" ]]; then
    echo "average-neighbor-distance sidecar not found: ${avg_neighbor_distance}" >&2
    exit 1
fi

metric() {
    local name="$1"
    local log="$2"
    awk -F= -v key="${name}" '$1 == key { value = $2 } END { print value }' "${log}"
}

mkdir -p "${result_root}"
printf 'topk\tmode\tsearch_mean_us\tdata_ready_wait_mean_us\tend_to_end_mean_us\ttp\tfp\tfn\twasted_io_ratio\ttotal_bytes_read\toracle_precompute_seconds\tlog\n' >"${summary}"

for topk in 10 20 50; do
    log="${result_root}/k${topk}_oracle.log"
    trace="${temporary_dir}/k${topk}_trace.bin"
    trace_index="${temporary_dir}/k${topk}_trace_index.tsv"

    echo "start topk=${topk} mode=oracle"
    command=(
        "${generator}"
        "${index}" "${queries}" "${trace}" "${trace_index}"
        "${topk}" "${ef_search}" "${batch_size}"
    )
    if [[ -n "${avg_neighbor_distance}" ]]; then
        command+=(--avg-neighbor-distance "${avg_neighbor_distance}")
    fi
    command+=(
        --prefetch on --oracle-prefetch
        --raw-data "${raw_data}" --payload-size "${payload_size}"
        --io-workers "${io_workers}" --io-queue-size "${io_queue_size}"
        --evict-file-cache
    )
    OMP_NUM_THREADS=1 "${command[@]}" 2>&1 | tee "${log}"

    printf '%s\toracle\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${topk}" \
        "$(metric raw_search_mean_us "${log}")" \
        "$(metric raw_data_ready_wait_mean_us "${log}")" \
        "$(metric raw_end_to_end_mean_us "${log}")" \
        "$(metric raw_prefetch_tp "${log}")" \
        "$(metric raw_prefetch_fp "${log}")" \
        "$(metric raw_prefetch_fn "${log}")" \
        "$(metric raw_wasted_io_ratio "${log}")" \
        "$(metric raw_total_bytes_read "${log}")" \
        "$(metric oracle_precompute_seconds "${log}")" \
        "${log}" >>"${summary}"
    echo "done topk=${topk} mode=oracle"
done

cat >"${result_root}/configuration.txt" <<EOF
mode=oracle
topk=10,20,50
ef_search=${ef_search}
batch_size=${batch_size}
io_workers=${io_workers}
io_queue_size=${io_queue_size}
payload_size=${payload_size}
generator=${generator}
index=${index}
queries=${queries}
raw_data=${raw_data}
onnx_model=disabled
oracle_model_overhead=zero
avg_neighbor_distance=${avg_neighbor_distance:-computed_at_startup}
omp_num_threads=1
evict_file_cache=true
oracle_precomputation_excluded_from_query_latency=true
EOF

echo "oracle benchmarks complete"
echo "summary=${summary}"
