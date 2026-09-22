# SIFT Query HNSW timestep labels

## Definition

`sift_query.fvecs`의 10,000개 query를 모두 사용했다. query ID는 파일의
0-based 행 번호이며 별도 sampling은 하지 않았다. 각 level-0 timestep에서
갱신된 `res`의 rank별 node가 검색 종료 후 최종 top-k에 포함되면 1, 아니면
0으로 기록했다.

```text
label(query, timestep, rank) =
    1  if timestep.node_ids[rank] is in final_top_k_ids
    0  otherwise
```

최종 top-k는 brute-force ground truth가 아니라 `efSearch=200`으로 수행한
해당 HNSW 검색의 최종 결과다. `res`가 k개로 채워진 뒤부터 timestep을
기록하므로 k=10과 k=20의 총 timestep 수는 조금 다를 수 있다.

## Parameters

```text
queries: 10,000
dimension: 128
index vectors: 100,000
metric: squared L2
efSearch: 200
check_relative_distance: true
bounded_queue: true
k: 10 and 20
```

## Outputs

| k | label binary | query index TSV |
|---:|---|---|
| 10 | `output/sift_query10000_k10_labels.bin` | `output/sift_query10000_k10_labels_index.tsv` |
| 20 | `output/sift_query10000_k20_labels.bin` | `output/sift_query10000_k20_labels_index.tsv` |

TSV 열은 다음과 같다.

```text
query_id, record_offset, timestep_count, label_count,
positive_count, negative_count
```

`record_offset`은 binary 파일 시작부터 해당 query record까지의 byte
offset이다.

## Binary format version 1

모든 정수와 float은 little-endian 고정 크기 값이다.

```text
[File header: 56 bytes]
magic               char[8] = "HNSWLBL1"
version             uint32 = 1
nq                  uint32
k                   uint32
ef_search           uint32
feature_count       uint32 = 6
reserved            uint32 = 0
total_timesteps     uint64
total_labels        uint64
positive_labels     uint64

[Repeated query record]
query_id            uint32
timestep_count      uint32
entry_distance      float32
final_top_k_ids     int32[k]
final_top_k_dists   float32[k]

  [Repeated timestep]
  pop_count          uint64
  expanded_node_id   int32
  search_progress    float32

    [Repeated rank]
    node_id           int32
    label             uint8
```

## Actual results

| k | timesteps | labels | positive | negative | binary size |
|---:|---:|---:|---:|---:|---:|
| 10 | 2,002,262 | 20,022,620 | 19,274,361 | 748,259 | 약 127 MiB |
| 20 | 2,001,779 | 40,035,580 | 38,243,143 | 1,792,437 | 약 224 MiB |

SHA-256:

```text
f997279e44d64f77c31ae0383e3ffc681b4565d55699509937beb6c11a23fdd8  sift_query10000_k10_labels.bin
ea724358449e3f84b072e4847fec5d88c0415c28a6de4f1b7a7b94423618093e  sift_query10000_k10_labels_index.tsv
efb9afe5dd38ee423966402fa11fd6a479b4303cd18b26a00be9f33d43bf6807  sift_query10000_k20_labels.bin
4b4176f2fdca7f7dbe2ec5993973b568265ca0357781a2a20207ee6546805792  sift_query10000_k20_labels_index.tsv
```

각 파일은 reader로 10,000개 query 전체를 다시 파싱하여 다음을 검증했다.

- query ID가 0부터 9,999까지 연속이다.
- 모든 label 값이 0 또는 1이다.
- 각 label이 node ID의 최종 top-k 포함 여부와 일치한다.
- header, TSV 및 실제 record의 timestep/label/positive 합계가 일치한다.
- 마지막 query 뒤에 trailing byte가 없다.

## Build and reproduce

```bash
cmake --build build-cpu --target generate_sift_query_labels --parallel 4

./build-cpu/demos/generate_sift_query_labels \
  ../../output/sift100k_hnsw_m32_efc200_efs200.index \
  ../../data/sift/sift_query.fvecs \
  ../../output/sift_query10000_k10_labels.bin \
  ../../output/sift_query10000_k10_labels_index.tsv \
  10 200 64
```

마지막 `64`는 메모리 사용량을 제한하기 위한 query batch 크기다. k=20은
출력 경로와 k 인자만 바꿔 동일하게 생성한다.

## Reader

```bash
python3 experiments/early-io-hnsw/read_sift_query_labels.py \
  ../../output/sift_query10000_k10_labels.bin \
  --show-first
```

reader는 query별 `node_ids`와 `labels`를 제공하며 파일 전체 검증도 수행한다.
