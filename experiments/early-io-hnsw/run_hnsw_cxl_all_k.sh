#!/usr/bin/env bash
# Server-side: rebuild current code, measure six runs, then draw one PNG.
# Usage: bash run_hnsw_cxl_all_k.sh [CPU_NODE=1] [MEMORY_NODE=2] [--dry-run]
set -euo pipefail

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
dry_run=false
positional=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=true ;;
        -h|--help)
            printf 'Usage: bash %s [CPU_NODE=1] [MEMORY_NODE=2] [--dry-run]\n' "$0"
            printf '%s\n' 'Environment: RESULT_ROOT, BUILD_DIR, PYTHON, BUILD_JOBS, MODEL_ROOT.'
            exit 0 ;;
        -*) fail "Unknown option: $argument" ;;
        *) positional+=("$argument") ;;
    esac
done
(( ${#positional[@]} <= 2 )) || fail 'Expected CPU_NODE and MEMORY_NODE only.'
cpu_node="${positional[0]:-1}"
memory_node="${positional[1]:-2}"
[[ "$cpu_node" =~ ^[0-9]+$ && "$memory_node" =~ ^[0-9]+$ ]] || fail 'Nodes must be non-negative integers.'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd -- "$repo_root"
build_dir="${BUILD_DIR:-${repo_root}/build-onnx-cpu}"
model_root="${MODEL_ROOT:-${repo_root}/model/sift10M/ef200/lstm_io}"
python="${PYTHON:-python3}"
plotter="${PLOT_SCRIPT:-${repo_root}/experiments/early-io-hnsw/python/plot_latency_breakdown_by_k.py}"
benchmark="${script_dir}/run_hnsw_cxl_benchmark.sh"
run_root="${RESULT_ROOT:-${repo_root}/workspace/results/async_stop}"
[[ "$run_root" == /* ]] || run_root="${repo_root}/${run_root}"
[[ ! -e "$run_root" ]] || fail "Result directory already exists; choose a new RESULT_ROOT: $run_root"
figure="${run_root}/figure/k10_k20_k50_latency_breakdown.png"

printf 'New result directory: %s\nCPU NUMA node: %s; memory NUMA node: %s\n' \
    "$run_root" "$cpu_node" "$memory_node"
if [[ "$dry_run" == false ]]; then
    [[ -f "$build_dir/CMakeCache.txt" ]] || fail "Configure the CPU ONNX build first: $build_dir"
    [[ -f "$plotter" && -f "$benchmark" ]] || fail 'Transfer the plotting script and use the updated repository.'
    for topk in 10 20 50; do
        [[ -r "$model_root/k${topk}/model.onnx" ]] || fail "Missing k${topk} model."
    done
    command -v numactl >/dev/null || fail 'numactl is required.'
    "$python" -c 'import matplotlib' || fail 'matplotlib is required in PYTHON.'
    numactl --cpunodebind "$cpu_node" --membind "$memory_node" true
    # Rebuild to avoid accidentally measuring an executable from before pull.
    cmake -S "$repo_root" -B "$build_dir"
    cmake --build "$build_dir" --target generate_sift_query_labels --parallel "${BUILD_JOBS:-4}"
    mkdir -p -- "$(dirname -- "$run_root")"
    mkdir -- "$run_root"
    {
        printf 'commit=%s\n' "$(git rev-parse HEAD)"
        printf 'cpu_node=%s\nmemory_node=%s\n' "$cpu_node" "$memory_node"
        printf 'model_root=%s\ngenerator=%s\n' "$model_root" "$build_dir/demos/generate_sift_query_labels"
        printf 'policy=use_rebuilt_current_source\n'
    } > "$run_root/configuration.txt"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
for topk in 10 20 50; do
    if [[ "$dry_run" == true ]]; then
        printf 'Would measure baseline + prefetch: k=%s -> %s/k%s_{baseline,prefetch}/\n' \
            "$topk" "$run_root" "$topk"
        continue
    fi
    RESULT_ROOT="$run_root" MEMORY_NODE="$memory_node" \
    GENERATOR_PATH="$build_dir/demos/generate_sift_query_labels" \
    ONNX_MODEL_PATH="$model_root/k${topk}/model.onnx" \
        bash "$benchmark" "$topk" "$cpu_node"
done

plot_command=("$python" "$plotter" --k 10 20 50 --result-root "$run_root"
    --baseline-dir k10_baseline k20_baseline k50_baseline
    --prefetch-dir k10_prefetch k20_prefetch k50_prefetch
    --output "$figure")
if [[ "$dry_run" == true ]]; then
    printf 'Would plot: '; printf '%q ' "${plot_command[@]}"; printf '\n'
    exit 0
fi

# Keep numeric means together, in ms, in addition to the six original logs.
summary="$run_root/summary_ms.tsv"
printf 'k\tmode\tqueries\tsearch_ms\tio_wait_ms\te2e_ms\tprediction_timesteps\n' > "$summary"
for topk in 10 20 50; do
    for mode in baseline prefetch; do
        awk -F= -v k="$topk" -v mode="$mode" '
            $1 == "raw_query_count" { n=$2 }
            $1 == "raw_search_mean_us" { s=$2; have_s=1 }
            $1 == "raw_data_ready_wait_mean_us" { w=$2; have_w=1 }
            $1 == "raw_end_to_end_mean_us" { e=$2; have_e=1 }
            $1 == "prediction_timesteps" { p=$2 }
            END {
                if (!have_s || !have_w || !have_e || n != 1000) exit 1
                printf "%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%s\n", k,mode,n,s/1000,w/1000,e/1000,p
            }
        ' "$run_root/k${topk}_${mode}/run.log" >> "$summary" ||
            fail 'Missing latency metrics or query count is not 1000; inspect run.log.'
    done
done
"${plot_command[@]}"
printf '\nComplete.\nResults: %s\nSummary: %s\nPNG: %s\n' "$run_root" "$summary" "$figure"
