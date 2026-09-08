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
  output/sift100k_internal_to_original.tsv
```

## 실제 구축 결과

- 샘플링 시간: 0.414초
- HNSW 구축 시간: 11.061초
- 파일 기록 시간: 0.118초
- `ntotal`: 100,000
- `max_level`: 3
- `entry_point`: 20,288
- 인덱스 크기: 약 75 MiB
- 매핑 크기: 약 1.3 MiB

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
