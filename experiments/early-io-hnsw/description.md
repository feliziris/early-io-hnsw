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
