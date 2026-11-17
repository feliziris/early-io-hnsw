벡터DB 에서 HNSW 기반으로 벡터 검색을 해서 top-k 를 선택하는데, 원래는 검색을 끝까지 해야 최종 top-k결과를 알 수 있어. 근데 나는 HNSW 검색 도중에 실시간으로 최종 top-k 결과에 들 벡터를 예측하는 걸 하고 있어. 2 layer-stacked lstm에 넣어줄 입력 데이터로 sift1M 데이터셋 중 10만 개의 벡터를 샘플링해서 벡터db에 구축할거야. 그리고 HNSW 검색 과정의 큐에서 가장 유사한 벡터 하나를 뽑아서 이웃들을 탐색하는 매 스텝마다 큐의 상위 top-k 후보들에 대해서 정적(static) 및 동적인 (dynamic) 런타임 피처를 수집하고, 각 후보들을 몇 가지의 feature로 나타내서 입력 데이터를 구성하고 있어. 그래서 나는 faiss에 lstm 모델을 삽입해서 런타임에 예측하는 걸 하고 싶어.

내가 사용하고 있는 피처는 다음과 같아.

# static feature
1. avg neigh dist: level-0에서 노드와 노드의 이웃들과의 평균 거리 (그래프 구축시 계산해놓을 피처)
# dynamic feature
2. ratio to entry: base layer의 entry point-query 거리와 노드-query 거리의 비율
3. ratio 1st: 큐에서의 1등-query와 노드-query의 거리 비율
4. rank norm: top-k 중 몇 등인지
5. persistence rate: 큐의 해당 rank에서 얼마동안 자리를 유지했는지
6. 검색 진행률(cnt_visit(큐에서 미방문 노드 중 해당 시점 가장 유사한 노드 pop 횟수 기준)/efSearch)

그래서 1번 피쳐는 hnsw 그래프 구축 시 한 노드에 대해서 노드와 이웃 간의 거리 계산을 통해 이웃들을 추가할 때 미리 offline에 계산해놓기 위한 소스코드를 삽입해야해.
그리고 나머지 2-6번 피쳐는 런타임 도중에 수집할 수 있는 소스코드를 삽입해야해.

top-k 설정에 따라 달라지겠지만, 예를 들어 top-10에 대해서 예측한다고 하면, 10\*6의 크기의 벡터로 hnsw 큐가 표현되는거야.
그래서 모델을 삽입해서 런타임에 예측하려면 수집한 데이터들을 저장해놓고 읽어올 자료구조 구현도 필요해.

현재 실험 계획은 top-10일 때와 top-20일 때를 비교하고 싶어. 그래서 k 파라미터에 따라 피처를 수집하는 개수가 달라질 수 있게 코드를 짜고 싶어.

## 학습 loss 종류

기존 I/O-aware Weighted BCE 외에, TP의 prefetch 결정을 앞당기기 위한
`Weighted BCE + TP Decision Delay Loss`를 추가 구현했다.
`train.py --loss io_aware_delay --delay-loss-weight 1.0 --prefetch-threshold 0.5`로
선택한다. 기존 loss는 기본값인 `--loss io_aware`로 계속 사용할 수 있다.

### 공통 정의

- `T`: 한 쿼리에서 loss 계산에 사용하는 유효 timestep 수. 배치 padding은 제외한다.
- `Q_t`: timestep `t`의 결과 큐(`res`)에 있는 유효한 top-k 노드 집합.
  다음에 확장할 노드를 관리하는 탐색 후보 큐와 구분한다.
- `P`: 해당 쿼리의 최종 HNSW top-k 노드 집합, `k = |P|`.
- `y_i`: 노드 `i`가 `P`에 포함되면 1, 아니면 0.
- `p_{i,t}`: timestep `t`에서 노드 `i`의 예측 확률.
- `sp_t = pop_count_t / efSearch`: 기존 검색 진행도 정의를 유지한다.
  `pop_count_t`는 level-0 탐색에서 후보를 pop한 누적 횟수이며,
  `t / T`로 바꾸지 않는다.
- `e_i`: 노드 `i`가 결과 큐의 top-k에 처음 등장한 timestep.
- `theta`: prefetch 결정 threshold. 학습되는 파라미터가 아니라 설정값이다.

### 1. 기존 I/O-aware Weighted BCE

라벨 1의 가중치는 1, 라벨 0의 가중치는 `1 + sp_t`로 유지한다.
새 조합 loss에서 사용할 BCE의 수식은 다음과 같다.

```text
L_BCE = (1/T) * sum_{t=1..T} (1/|Q_t|) * sum_{i in Q_t}
        [ -y_i * log(p_{i,t})
          -(1 + sp_t) * (1 - y_i) * log(1 - p_{i,t}) ]
```

위 수식은 timestep마다 유효한 노드의 loss를 평균낸 뒤 timestep 평균을 낸다.
현재 `train.py`의 `io_aware_loss()`는 top-k loss를 합산한 뒤 배치 전체의
유효 timestep 수로만 나누므로, 위 수식과 정규화 방식에 차이가 있다.
새 `io_aware_delay` 옵션은 위 수식의 정규화를 적용하며, 기존 `io_aware`
옵션의 정규화는 유지한다. `1 + sp_t` 가중치 자체는 변경하지 않는다.

### 2. TP Decision Delay Loss

노드별로 최초 등장 이후의 예측 확률 최댓값을 유지한다.
노드의 rank가 바뀌어도 같은 노드 ID의 이력을 이어서 사용한다.

```text
m_{i,t} = max_{e_i <= u <= t} p_{i,u}

L_Delay = (1/(k*T)) * sum_{i in P} sum_{t=e_i..T}
          [ max(0, theta - m_{i,t}) / theta ]
```

- 최종 top-k에 포함되는 노드만 Delay Loss에 포함한다.
- 최초 등장 전에는 패널티를 부여하지 않는다.
- 아직 threshold에 도달하지 않았다면, 부족한 확률에 비례하는 패널티를
  timestep마다 부여한다.
- 한 번 threshold에 도달하면 이후 확률이 낮아져도 running maximum은
  threshold 이상이므로 추가 패널티가 없다.
- `theta > 0`이어야 하며 padding timestep은 합계와 `T`에서 제외한다.

이 항은 실제 threshold 도달 시간의 직접적인 평균이 아니라,
TP의 빠른 결정을 유도하기 위한 미분 가능한 패널티다.

### 3. 최종 조합 loss

```text
L_total = L_BCE + lambda_D * L_Delay
```

`lambda_D`는 Delay Loss의 상대 가중치이며 실험 설정으로 정한다.
쿼리별로 정의된 loss를 계산하고 배치에서 평균내는 형태의 설계다.
이 loss는 클래스 빈도에 따른 0/1 불균형 보정을 별도로 포함하지 않는다.

현재 데이터 로더는 마지막 timestep을 제외하므로, 해당 정책을 유지하면
`T`와 노드별 이력도 마지막 timestep을 제외한 학습 구간을 기준으로 한다.
