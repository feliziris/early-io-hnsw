#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
faiss_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${faiss_root}/.." && pwd)"
project_root="$(cd "${workspace_root}/.." && pwd)"

repetitions="${REPETITIONS:-3}"
comparison_mode="${COMPARISON_MODE:-oracle}"
ef_search="${EF_SEARCH:-200}"
batch_size="${BATCH_SIZE:-1}"
io_workers="${IO_WORKERS:-4}"
io_queue_size="${IO_QUEUE_SIZE:-4096}"
payload_size="${PAYLOAD_SIZE:-4096}"
build_dir="${BUILD_DIR:-${project_root}/build-onnx}"
generator="${build_dir}/demos/generate_sift_query_labels"
index="${INDEX_PATH:?set INDEX_PATH}"
queries="${QUERY_PATH:?set QUERY_PATH}"
raw_data="${RAW_DATA_PATH:?set RAW_DATA_PATH}"
avg_neighbor_distance="${AVG_NEIGHBOR_DISTANCE_PATH:?set AVG_NEIGHBOR_DISTANCE_PATH}"
prefetch_schedule_dir="${PREFETCH_SCHEDULE_DIR:-}"
result_root="${RESULT_ROOT:-${workspace_root}/results/hnsw_baseline_oracle_paired/$(date +%Y%m%d_%H%M%S)}"
temporary_dir="$(mktemp -d /tmp/hnsw-baseline-oracle-paired.XXXXXX)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

if [[ "${comparison_mode}" != "oracle" && "${comparison_mode}" != "replay" ]]; then
    echo "COMPARISON_MODE must be oracle or replay" >&2
    exit 2
fi
if [[ "${comparison_mode}" == "replay" && -z "${prefetch_schedule_dir}" ]]; then
    echo "PREFETCH_SCHEDULE_DIR is required for replay mode" >&2
    exit 2
fi

metric() {
    local name="$1"
    local log="$2"
    awk -F= -v key="${name}" '$1 == key { value = $2 } END { print value }' "${log}"
}

mkdir -p "${result_root}"
runs="${result_root}/paired_runs.tsv"
means="${result_root}/paired_mean.tsv"
printf 'round\ttopk\tmode\tsearch_mean_us\tdata_ready_wait_mean_us\tend_to_end_mean_us\ttp\tfp\tfn\twasted_io_ratio\tlog\n' >"${runs}"

run_one() {
    local round="$1"
    local topk="$2"
    local mode="$3"
    local log="${result_root}/round_${round}_k${topk}_${mode}.log"
    local trace="${temporary_dir}/round_${round}_k${topk}_${mode}.bin"
    local trace_index="${temporary_dir}/round_${round}_k${topk}_${mode}.tsv"
    local command=(
        "${generator}" "${index}" "${queries}" "${trace}" "${trace_index}"
        "${topk}" "${ef_search}" "${batch_size}"
        --avg-neighbor-distance "${avg_neighbor_distance}"
        --raw-data "${raw_data}" --payload-size "${payload_size}"
        --io-workers "${io_workers}" --io-queue-size "${io_queue_size}"
        --evict-file-cache
    )
    if [[ "${mode}" == "baseline" ]]; then
        command+=(--warmup-search --prefetch off)
    elif [[ "${mode}" == "oracle" ]]; then
        command+=(--prefetch on --oracle-prefetch)
    else
        command+=(
            --warmup-search
            --prefetch on
            --prefetch-replay-in
            "${prefetch_schedule_dir}/k${topk}_threshold05.rpl"
        )
    fi

    echo "start round=${round} topk=${topk} mode=${mode}"
    OMP_NUM_THREADS=1 "${command[@]}" 2>&1 | tee "${log}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${round}" "${topk}" "${mode}" \
        "$(metric raw_search_mean_us "${log}")" \
        "$(metric raw_data_ready_wait_mean_us "${log}")" \
        "$(metric raw_end_to_end_mean_us "${log}")" \
        "$(metric raw_prefetch_tp "${log}")" \
        "$(metric raw_prefetch_fp "${log}")" \
        "$(metric raw_prefetch_fn "${log}")" \
        "$(metric raw_wasted_io_ratio "${log}")" "${log}" >>"${runs}"
}

for round in $(seq 1 "${repetitions}"); do
    for topk in 10 20 50; do
        if (( round % 2 == 1 )); then
            run_one "${round}" "${topk}" baseline
            run_one "${round}" "${topk}" "${comparison_mode}"
        else
            run_one "${round}" "${topk}" "${comparison_mode}"
            run_one "${round}" "${topk}" baseline
        fi
    done
done

printf 'topk\tmode\tsearch_mean_us\tdata_ready_wait_mean_us\tend_to_end_mean_us\trepetitions\n' >"${means}"
for topk in 10 20 50; do
    for mode in baseline "${comparison_mode}"; do
        awk -F'\t' -v k="${topk}" -v m="${mode}" '
            NR > 1 && $2 == k && $3 == m {
                search += $4; wait += $5; total += $6; count++
            }
            END {
                printf "%s\t%s\t%.6f\t%.6f\t%.6f\t%d\n",
                    k, m, search / count, wait / count, total / count, count
            }
        ' "${runs}" >>"${means}"
    done
done

cat >"${result_root}/configuration.txt" <<EOF
comparison=paired_baseline_${comparison_mode}
repetitions=${repetitions}
execution_order=alternated_by_round
topk=10,20,50
ef_search=${ef_search}
batch_size=${batch_size}
io_workers=${io_workers}
io_queue_size=${io_queue_size}
payload_size=${payload_size}
index=${index}
queries=${queries}
raw_data=${raw_data}
avg_neighbor_distance=${avg_neighbor_distance}
omp_num_threads=1
evict_file_cache=true
baseline_untimed_warmup=true
oracle_precomputation_excluded=true
onnx_model=disabled
prefetch_schedule_dir=${prefetch_schedule_dir}
EOF

echo "paired benchmark complete"
echo "runs=${runs}"
echo "means=${means}"
