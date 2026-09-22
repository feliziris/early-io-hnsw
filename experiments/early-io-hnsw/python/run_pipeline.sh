#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
faiss_dir="$(cd "${project_root}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
generator="${faiss_dir}/build-cpu/demos/generate_sift_query_labels"
index="${project_root}/output/sift100k_hnsw_m32_efc200_efs200.index"
queries="${project_root}/data/sift/sift_query.fvecs"
output_dir="${project_root}/output"
model_dir="${project_root}/model/sift/lstm_io"

cmake --build "${faiss_dir}/build-cpu" --target generate_sift_query_labels -j2
mkdir -p "${output_dir}" "${model_dir}/k10" "${model_dir}/k20" "${model_dir}/k50"

for k in 10 20 50; do
    trace="${output_dir}/sift_query10000_k${k}_trace.bin"
    trace_index="${output_dir}/sift_query10000_k${k}_trace_index.tsv"
    hdf5="${output_dir}/sift_query10000_k${k}.h5"

    OMP_NUM_THREADS=12 "${generator}" \
        "${index}" "${queries}" "${trace}" "${trace_index}" \
        "${k}" 200 100

    "${python_bin}" "${project_root}/python/convert_trace_to_hdf5.py" \
        --trace "${trace}" --output "${hdf5}" --seed 42 --train-ratio 0.8

    "${python_bin}" "${project_root}/python/train.py" \
        --data "${hdf5}" --output "${model_dir}/k${k}" \
        --epochs 30 --batch-size 32 --workers 0 --seed 42 --device cpu
done
