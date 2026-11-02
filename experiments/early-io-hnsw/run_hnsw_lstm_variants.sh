#!/usr/bin/env bash
# Nine sequential measurements with a single fresh baseline for each k.
set -euo pipefail
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd -- "$repo_root"
experiment_root="${EXPERIMENT_ROOT:-$repo_root}"
build_dir="${BUILD_DIR:-${experiment_root}/build-onnx-cpu}"
index="${INDEX_PATH:-${experiment_root}/data/index/bigann_base10M_hnsw_M32_efC128.faiss}"
queries="${QUERY_PATH:-${experiment_root}/output/sift10M/bigann_query_test1000_seed42.fvecs}"
sidecar="${AVG_NEIGHBOR_DISTANCE_PATH:-${experiment_root}/data/index/bigann_base10M_hnsw_M32_efC128.level0_avg_neighbor_distance.bin}"
raw="${RAW_DATA_PATH:-${experiment_root}/workspace/raw_data/payload4k_10m.bin}"
model_root="${MODEL_ROOT:-${experiment_root}/model/sift10M/ef200}"
result_root="${RESULT_ROOT:-${experiment_root}/workspace/results/lstm_layer_comparison}"
python="${PYTHON:-python3}"
cpu_node="${CPU_NODE:-0}"
memory_node="${MEMORY_NODE:-0}"
[[ ! -e "$result_root" ]] || fail "Result directory exists: $result_root"
for file in "$index" "$queries" "$sidecar" "$raw" "$build_dir/CMakeCache.txt"; do
    [[ -r "$file" ]] || fail "Missing input: $file"
done
for k in 10 20 50; do
    for variant in lstm_io lstm_io_1layer_h128; do
        [[ -r "$model_root/$variant/k$k/model.onnx" ]] || fail "Missing model: $variant/k$k"
    done
done
"$python" -c 'import matplotlib'
numactl --cpunodebind "$cpu_node" --membind "$memory_node" true
cmake -S "$repo_root" -B "$build_dir"
cmake --build "$build_dir" --target generate_sift_query_labels --parallel "${BUILD_JOBS:-4}"
mkdir -p -- "$(dirname -- "$result_root")"
mkdir -- "$result_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
    printf 'commit=%s\n' "$(git rev-parse HEAD)"
    printf 'cpu_node=%s\nmemory_node=%s\n' "$cpu_node" "$memory_node"
    printf 'onnx_provider=cpu\nonnx_inference=async\nefSearch=200\nthreshold=0.5\n'
    printf 'payload_bytes=4096\nio_workers=4\nio_queue_size=4096\n'
    printf 'omp_num_threads=1\nopenblas_num_threads=1\nrepetitions_per_mode=1\n'
    printf 'prediction_policy=stop_at_graph_end\noutlier_policy=no_samples_removed\n'
    printf 'e2e_definition=search_start_to_final_topk_payload_ready\n'
    printf 'post_data_ready_inference_cleanup=excluded_from_raw_e2e_but_joined_before_search_return\n'
    printf 'model_root=%s\n' "$model_root"
    uname -srm
} > "$result_root/configuration.txt"
summary="$result_root/summary_ms.tsv"
printf 'k\tmode\tqueries\tsearch_ms\tio_wait_ms\te2e_ms\tprediction_timesteps\ttp\tfp\tfn\twasted_io_ratio\n' > "$summary"
for k in 10 20 50; do
    for mode in baseline lstm_2layer lstm_1layer; do
        directory="$result_root/k${k}_${mode}"
        mkdir -- "$directory"
        command=(numactl --cpunodebind "$cpu_node" --membind "$memory_node"
            "$build_dir/demos/generate_sift_query_labels" "$index" "$queries"
            "$directory/trace.bin" "$directory/trace_index.tsv" "$k" 200 1
            --avg-neighbor-distance "$sidecar" --raw-data "$raw"
            --payload-size 4096 --io-workers 4 --io-queue-size 4096
            --evict-file-cache --warmup-search)
        if [[ "$mode" == baseline ]]; then
            command+=(--prefetch off)
        else
            variant=lstm_io
            [[ "$mode" != lstm_1layer ]] || variant=lstm_io_1layer_h128
            command+=(--prefetch on --onnx-model "$model_root/$variant/k$k/model.onnx"
                --onnx-provider cpu --onnx-inference async --threshold 0.5)
        fi
        {
            printf 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
            printf '%q ' "${command[@]}"; printf '\n'
        } > "$directory/command.sh"
        printf '\nRunning k=%s mode=%s\n' "$k" "$mode"
        "${command[@]}" 2>&1 | tee "$directory/run.log"
        awk -F= -v k="$k" -v mode="$mode" '
            $1=="raw_query_count" {n=$2}
            $1=="raw_search_mean_us" {s=$2; hs=1}
            $1=="raw_data_ready_wait_mean_us" {w=$2; hw=1}
            $1=="raw_end_to_end_mean_us" {e=$2; he=1}
            $1=="prediction_timesteps" {p=$2}
            $1=="raw_prefetch_tp" {tp=$2}
            $1=="raw_prefetch_fp" {fp=$2}
            $1=="raw_prefetch_fn" {fn=$2}
            $1=="raw_wasted_io_ratio" {waste=$2}
            END {
                if (n!=1000 || !hs || !hw || !he) exit 1
                printf "%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%s\t%s\t%s\t%s\t%s\n",k,mode,n,s/1000,w/1000,e/1000,p,tp,fp,fn,waste
            }
        ' "$directory/run.log" >> "$summary" || fail "Invalid metrics in $directory/run.log"
    done
done
"$python" "$script_dir/python/plot_lstm_variants.py" --summary "$summary" \
    --output "$result_root/figure/k10_k20_k50_latency_breakdown.png"
printf '\nAll nine measurements complete: %s\n' "$result_root"
