# SIFT 100K HNSW 인덱스 구축 설명

## 개요

SIFT1M의 base 벡터 1,000,000개 중 100,000개를 고정 seed로 중복 없이
샘플링하고, 해당 벡터로 FAISS `IndexHNSWFlat` 인덱스를 구축했다.

- 구축일: 2026-09-08
- 입력 데이터: `sift_base.fvecs`
- 입력 벡터 수: 1,000,000
- 벡터 차원: 128
- 거리 척도: L2 (`faiss::METRIC_L2`)
- 샘플 벡터 수: 100,000

## 샘플링 및 데이터셋 구성

1. SIFT 원본 ID를 파일의 0-based 행 번호로 정의했다. `.fvecs` 파일에는
   별도 ID가 없으므로 첫 벡터는 ID 0, 마지막 벡터는 ID 999,999이다.
2. `[0, 1, ..., 999999]` ID 배열을 생성했다.
3. C++ `std::mt19937`을 seed 42로 초기화했다.
4. `std::shuffle`로 배열을 섞고 앞의 100,000개를 선택했다. 복원추출이
   아니므로 선택된 원본 ID는 중복되지 않는다.
5. 선택된 ID 순서를 샘플 데이터셋 순서이자 HNSW 삽입 순서로 사용했다.
   파일 읽기는 원본 ID 순으로 수행하되 읽은 벡터를 원래 샘플 슬롯에
   배치했으므로 HNSW 삽입 순서는 바뀌지 않는다.

샘플링 결과의 첫 다섯 원본 ID는 다음과 같다.

```text
270578, 623089, 860023, 332895, 294746
```

구축 프로그램은 `internal_id`, `original_id` 열을 가진 TSV 매핑을 함께
생성한다. HNSW 검색이 반환한 내부 ID를 TSV의 원본 ID로 변환하면 SIFT
원본 행을 찾을 수 있다.

## HNSW 구축 파라미터

| 항목 | 값 |
|---|---:|
| 인덱스 종류 | `faiss::IndexHNSWFlat` |
| 거리 척도 | L2 |
| M | 32 |
| efConstruction | 200 |
| efSearch | 200 |
| deterministic build | `true` |

`faiss::hnsw_deterministic_build = true`를 `index.add()` 전에 설정했다.
이는 병렬 스케줄 순서와 무관하게 동일한 입력과 설정으로 재현 가능한
그래프를 만들기 위한 구축 옵션이며 검색 정확도를 조절하는 값은 아니다.
`efSearch=200`은 직렬화된 인덱스에 저장되며 검색 시 덮어쓸 수 있다.

## Level-0 평균 이웃 거리 피처

최종 그래프 구축 후 `IndexHNSW::compute_level0_avg_neighbor_distance()`를
호출한다. 각 노드의 유효한 level-0 이웃만 대상으로 native metric 거리의
산술평균을 계산한다. 현재 L2 인덱스에서 이 값은 squared L2 distance다.

계산값은 `IndexHNSW::level0_avg_neighbor_distance` 벡터에서 내부 ID로
조회한다. 기존 FAISS 인덱스 직렬화 호환성을 유지하기 위해 인덱스 파일에는
넣지 않고 별도 TSV sidecar로 저장한다.

## 검색 중 동적 피처 수집

`SearchParametersHNSW::feature_collector`에 `HNSWFeatureCollector`를
지정하면 level-0 검색에서 노드 하나를 pop하여 이웃을 모두 평가하고
`res` heap을 갱신한 직후마다 현재 best-so-far top-k를 수집한다. 원본 heap은
변경하지 않고 유효한 heap slot 번호만 거리와 ID 순서로 정렬한다.

각 timestep은 확장한 노드 ID, pop 횟수, top-k 내부 ID와 query 거리, 그리고
rank-major 형태의 `k * 6` 피처를 저장한다. slot별 순서는 다음과 같다.

```text
ratio_to_entry, ratio_to_first, search_progress,
rank_norm, persistence_rate, avg_neigh_dist
```

- `ratio_to_entry = d_node / d_entry`
- `ratio_to_first = d_first / d_node`
- `search_progress = pop_count / efSearch` (clamp하지 않음)
- `rank_norm = rank / (k - 1)`
- `persistence_rate`: 같은 노드가 동일 rank를 연속 유지한 횟수 / timestep 수
- `avg_neigh_dist`: 사전 계산한 level-0 평균 이웃 거리

검색 종료 시 최종 top-k ID와 거리도 query trace에 저장한다. 현재 collector는
`METRIC_L2`, `bounded_queue=true`, non-Panorama 검색을 지원하며 결과는
메모리에 유지한다. 디스크 직렬화 형식은 아직 추가하지 않았다.

## 검색 루프 내부 ONNX LSTM 추론

`FAISS_ENABLE_ONNX_RUNTIME=ON`으로 빌드하면 `HNSWOnnxModel`을 통해
stateful ONNX LSTM을 로드할 수 있다. `SearchParametersHNSW::onnx_model`에
모델을 지정하고 `onnx_threshold`를 설정하면 level-0 결과 큐가 갱신된 매
timestep마다 6개 피처로 추론한다. hidden/cell state는 query마다 0으로
초기화되고 query 내부에서만 다음 timestep으로 전달된다. batch/OpenMP 검색
시에도 query state는 서로 공유하지 않는다.

```cpp
faiss::SearchParametersHNSW params;
params.efSearch = 200;
faiss::HNSWOnnxModel model(model_path);
params.onnx_model = &model;
params.onnx_threshold = 0.5f;
```

모델을 지정하지 않으면 ONNX 추론을 전혀 실행하지 않아 기존 baseline 검색
경로를 유지한다. `HNSWFeatureTimestep::prediction_probabilities`와
`prefetch_decisions`에서 timestep별 결과를 검사할 수 있다.

## 검색 루프 내부 raw payload prefetch

`HNSWRawDataPrefetcher`는 raw 파일을 한 번 `mmap`하고 bounded queue와 worker
thread로 개별 512 KiB payload를 읽는다. `prefetch_enabled=true`이면 매
timestep의 ONNX 결정 직후 해당 내부 node ID를 제출하며, 같은 query에서 이미
제출한 node는 다시 제출하지 않는다. 검색 종료 시 최종 top-k 중 제출한 요청은
완료를 기다리고, 놓친 node만 검색 thread에서 동기 읽는다.

`prefetch_enabled=false`인 baseline은 ONNX 추론이나 early 제출은 수행하지
않지만, 검색 종료 후 최종 top-k payload를 동일한 worker queue에 한꺼번에
제출한다. prefetch가 놓친 최종 node도 검색 종료 후 같은 방식으로 제출한다.
따라서 baseline과 prefetch는 같은 worker 수와 I/O 병렬성을 사용하며, 차이는
읽기 요청을 검색 종료 후 제출하는지 검색 도중 미리 제출하는지다. 두 모드의
end-to-end 경계는 모두 "HNSW 검색 시작부터 최종 top-k payload 준비 완료"다.

```cpp
faiss::HNSWRawDataPrefetcher payloads(
        raw_path, 512 * 1024, 4, 4096, false);
params.raw_data_prefetcher = &payloads;
params.prefetch_enabled = true;
```

실행 예시는 다음과 같다. `--evict-file-cache`는 실행 직전 해당 raw 파일에
`POSIX_FADV_DONTNEED` 힌트를 적용한다.

```bash
# baseline
./build-onnx/demos/generate_sift_query_labels \
  INDEX QUERY TRACE TRACE_INDEX 10 200 10 \
  --prefetch off --raw-data payload512k.bin \
  --payload-size 524288 --io-workers 4 --io-queue-size 4096

# ONNX + asynchronous prefetch
./build-onnx/demos/generate_sift_query_labels \
  INDEX QUERY TRACE TRACE_INDEX 10 200 10 \
  --onnx-model model.onnx --threshold 0.5 \
  --prefetch on --raw-data payload512k.bin \
  --payload-size 524288 --io-workers 4 --io-queue-size 4096

# IDEAL ORACLE EVALUATION ONLY: 모델 비용 0, 정답 기반 비교군
./build-onnx/demos/generate_sift_query_labels \
  INDEX QUERY TRACE TRACE_INDEX 10 200 10 \
  --prefetch on --oracle-prefetch --raw-data payload512k.bin \
  --payload-size 524288 --io-workers 4 --io-queue-size 4096
```

`--oracle-prefetch`는 실제 시스템에 배포할 수 있는 정책이 아니라 모델 실행
비용과 분류 오류를 모두 제거한 이상적인 평가 전용 모드다. 프로그램은 먼저
ONNX와 raw I/O 없이
동일한 HNSW 파라미터로 모든 query의 최종 top-k를 계산한다. 이 시간은
`oracle_precompute_seconds`로 별도 출력되며 아래의 query latency에는 포함되지
않는다. 측정 검색에서는 ONNX 모델을 생성하거나 실행하지 않는다. 각 timestep의
결과 queue에서 미리 계산한 최종 top-k 노드를 발견하는 즉시 I/O를 제출하고,
동일한 노드는 한 번만 요청한다. 따라서 `FP=0`, 모델 추론 timestep은 0이며,
검색 traversal과 결과 queue 갱신 방식은 일반 검색과 동일하다.

일반 prefetch와 oracle의 end-to-end 차이에는 모델 실행 비용, 잘못된 예측,
결정 지연의 영향이 함께 포함된다. I/O가 서로 겹쳐 실행되므로 각 요인의
지연시간을 단순 합산해서 정확히 분해할 수는 없다.

출력에는 query별 `search`, `data_ready_wait`, `end_to_end` latency 분포와
비동기 queue wait, 실제 mmap read, decision-to-completion latency가 포함된다.
`wasted_io_ratio`는 중복 제거된 node 기준으로
`prefetch_fp / (prefetch_tp + prefetch_fn)`이다.

전체 query를 batch size 1로 baseline과 prefetch 순서로 반복 실행하려면 다음
스크립트를 사용한다. 기본값은 1회 반복, k=10, batch size=1, threshold=0.5다.
기본 query 입력은 HDF5의 seed-42 `test_query_ids`로 추출한 2,000개 test
query인 `output/sift_query_test_seed42.fvecs`다.

최초 한 번 test query 파일과 원본 query ID 매핑을 생성한다.

```bash
python experiments/early-io-hnsw/python/extract_test_queries.py \
  --source ../data/sift/sift_query.fvecs \
  --hdf5 ../output/sift_query10000_k10.h5 \
  --output ../output/sift_query_test_seed42.fvecs \
  --mapping ../output/sift_query_test_seed42_mapping.tsv
```

```bash
./experiments/early-io-hnsw/run_hnsw_prefetch_benchmark.sh

# repetitions, top-k, batch-size, threshold
./experiments/early-io-hnsw/run_hnsw_prefetch_benchmark.sh 5 20 1 0.6
```

각 실행에 `--evict-file-cache`가 적용되며 로그, 설정, 요약 TSV는
`workspace/results/hnsw_live_prefetch/<실행 시각>/`에 저장된다. trace 파일은
측정 중 임시 디렉터리에 기록되고 실행 종료 시 제거된다.

k=10, 20, 50을 차례로 모두 실행하려면 다음 wrapper를 사용한다.

```bash
./experiments/early-io-hnsw/run_all_hnsw_prefetch_benchmarks.sh
```

ONNX Runtime 개발 헤더와 라이브러리 위치를 명시해 빌드한다.

```bash
cmake -S . -B build-onnx \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_ONNX_RUNTIME=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so
cmake --build build-onnx --target generate_sift_query_labels -j2
```

`generate_sift_query_labels`의 기존 인자 뒤에 모델 경로와 threshold를
선택적으로 전달할 수 있다.

```bash
./build-onnx/demos/generate_sift_query_labels \
  INDEX QUERY TRACE TRACE_INDEX 10 200 10 \
  --onnx-model MODEL_ONNX --threshold 0.5
```

## 평균 이웃 거리 오프라인 저장 및 로드

`avg_neigh_dist`는 쿼리가 아니라 완성된 HNSW 그래프에 종속되는 값이다.
따라서 큰 인덱스에서는 검색 실행마다 계산하지 않고 인덱스당 한 번 계산해
바이너리 sidecar로 저장한다.

```bash
cmake --build build-cpu \
  --target precompute_hnsw_avg_neighbor_distance --parallel 4

./build-cpu/demos/precompute_hnsw_avg_neighbor_distance \
  INDEX.faiss INDEX.level0_avg_neighbor_distance.bin
```

이후 검색 및 trace 생성 시 파일을 로드한다.

```bash
./build-onnx/demos/generate_sift_query_labels \
  INDEX.faiss QUERY.fvecs TRACE.bin TRACE_INDEX.tsv 10 200 1 \
  --avg-neighbor-distance INDEX.level0_avg_neighbor_distance.bin \
  --onnx-model MODEL.onnx --threshold 0.5
```

## ONNX Runtime CPU/CUDA 실행 선택

검색 루프의 LSTM 추론 provider는 실행 시 선택한다. `gpu`는 `cuda`의
별칭이며, CUDA를 명시했는데 `CUDAExecutionProvider`를 사용할 수 없으면
CPU로 조용히 대체하지 않고 오류를 낸다.

```bash
# CPU
./build-onnx/demos/generate_sift_query_labels \
  INDEX.faiss QUERY.fvecs TRACE.bin TRACE_INDEX.tsv 10 200 1 \
  --onnx-model MODEL.onnx --onnx-provider cpu

# CUDA device 0
./build-onnx/demos/generate_sift_query_labels \
  INDEX.faiss QUERY.fvecs TRACE.bin TRACE_INDEX.tsv 10 200 1 \
  --onnx-model MODEL.onnx --onnx-provider cuda --onnx-device-id 0
```

벤치마크 스크립트에서는 `ONNX_PROVIDER`와 `ONNX_DEVICE_ID` 환경변수를
사용한다.

```bash
ONNX_PROVIDER=cuda ONNX_DEVICE_ID=0 \
  experiments/early-io-hnsw/run_hnsw_prefetch_benchmark.sh 1 10 1 0.5
```

CUDA 경로는 쿼리별 LSTM hidden/cell state를 GPU 메모리의 두 버퍼에서
번갈아 사용한다. 따라서 매 타임스텝마다 recurrent state를 CPU로 다시
복사하지 않는다. 현재 HNSW feature는 CPU에서 생성되고, prefetch 결정을
위한 확률은 매 타임스텝 CPU로 반환되므로 동기화 비용은 남아 있다.

Jetson CUDA 12.6 환경에는 `onnxruntime-gpu 1.24.0`을 설치했다. k=10
모델의 500 timestep 검증에서 CPU/CUDA 출력의 최대 절대 오차는 1e-4
이하였으며 검증을 통과했다. 동일한 1,000개 쿼리, `efSearch=200`,
threshold 0.5 조건의 평균 결과는 다음과 같다.

| provider | search (ms/query) | I/O wait (ms/query) | end-to-end (ms/query) |
|---|---:|---:|---:|
| CPU | 28.424166 | 0.003216 | 28.427382 |
| CUDA | 204.231308 | 0.004390 | 204.235698 |

CUDA의 end-to-end latency는 CPU의 약 7.18배였다. 배치 1의 작은 LSTM을
약 200 timestep 동안 개별 실행하며 각 timestep에 확률을 동기화하는 현재
구조에서는 GPU kernel launch와 CPU/GPU 동기화 비용이 계산 이득보다 크다.
원본 로그와 통합 결과는
`workspace/results/hnsw_sift10m_4k/20261007_k10_cpu_vs_cuda/`에 있다.

sidecar에는 값뿐 아니라 `ntotal`, 차원, metric, 그래프 배열 크기,
entry point와 max level이 함께 저장된다. 로드한 인덱스와 메타데이터가
다르면 사용하지 않고 오류를 낸다. 그래프에 `add()` 또는 `reset()`을 한
뒤에는 기존 sidecar를 재사용하지 말고 다시 생성해야 한다.

## 코드와 실행 방법

- 구축 소스: `demos/build_sift100k_hnsw.cpp`
- CMake 등록: `demos/CMakeLists.txt`
- CMake 타깃: `build_sift100k_hnsw`

CPU-only Release 구성 예시는 다음과 같다.

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_OPT_LEVEL=generic \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_EXTRAS=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-cpu --target build_sift100k_hnsw --parallel 12
```

실행 시 입력, 인덱스 출력, 매핑 출력 경로를 차례로 전달한다.

```bash
mkdir -p output
./build-cpu/demos/build_sift100k_hnsw \
  /path/to/sift_base.fvecs \
  output/sift100k_hnsw_m32_efc200_efs200.index \
  output/sift100k_internal_to_original.tsv \
  output/sift100k_level0_avg_neighbor_distance.tsv
```

## 실제 구축 결과

- 샘플링 시간: 0.360초
- HNSW 구축 시간: 8.848초
- level-0 평균 이웃 거리 계산 시간: 0.080초
- 파일 기록 시간: 0.293초
- `ntotal`: 100,000
- `max_level`: 3
- `entry_point`: 20,288
- 인덱스 크기: 약 75 MiB
- 매핑 크기: 약 1.3 MiB
- 평균 이웃 거리 sidecar 크기: 약 1.9 MiB
- level-0 이웃 수: 최소 2개, 최대 64개, 평균 27.20789개
- 고립 노드 수: 0개

매핑 파일은 헤더를 제외하고 정확히 100,000행이며 다음을 검증했다.

- 내부 ID가 0부터 99,999까지 순서대로 존재한다.
- 원본 ID가 모두 0부터 999,999 범위 안에 있다.
- 원본 ID 100,000개에 중복이 없다.

## 저장소에 포함하지 않는 파일

SIFT 원본 데이터, CMake 빌드 디렉터리, 생성된 HNSW 인덱스는 Git 저장소에
포함하지 않는다. 데이터셋은 별도로 준비하고 위 프로그램으로 결과를 다시
생성한다.

`std::shuffle`의 정확한 순열은 C++ 표준 라이브러리 구현에 영향을 받을 수
있다. 다른 툴체인에서도 완전히 같은 표본이 필요하면 최초 실행에서 생성한
TSV 매핑을 원본 ID 목록으로 보존해야 한다.

## 탐색 종료 시 비동기 예측 중단

현재 live 비동기 검색은 그래프 탐색이 끝나면 아직 시작하지 않은 snapshot을
버리고, 최종 top-k 중 이미 제출한 요청을 재사용하며 나머지를 즉시 같은 I/O
worker queue에 제출한다. 최종 I/O 요청을 제출하기 전에 예측 backlog를
기다리지 않는다. 이미 실행 중인 CPU ONNX 호출이나 CUDA 작업은 강제 취소하지
않으며, payload I/O와 겹쳐 마무리한 뒤 반환 전에 자원을 정리한다. 따라서
payload 준비 완료 시각과 실제 search() 반환 시각은 다를 수 있다.

취소된 timestep의 prediction/decision 배열은 비어 있다. raw payload reader가
없는 trace 수집·검증 검색은 기존처럼 모든 timestep의 예측을 완료한다.
