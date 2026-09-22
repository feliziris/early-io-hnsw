# Jetson HNSW LSTM training package

The three HDF5 files already contain the six features, labels, metadata and
the deterministic seed-42 query split (8,000 train / 2,000 test).

## Files

```text
data/sift_query10000_k10.h5
data/sift_query10000_k20.h5
data/sift_query10000_k50.h5
python/train.py
python/dataset.py
python/model.py
python/schema.py
python/requirements.txt
```

## Training

Run commands from the extracted `jetson_train` directory. Use a PyTorch build
that is compatible with the Jetson/JetPack version installed on the machine.

```bash
python python/train.py --data data/sift_query10000_k10.h5 \
  --output model/sift/lstm_io/k10 --epochs 30 --batch-size 32 \
  --workers 0 --seed 42 --device auto

python python/train.py --data data/sift_query10000_k20.h5 \
  --output model/sift/lstm_io/k20 --epochs 30 --batch-size 32 \
  --workers 0 --seed 42 --device auto

python python/train.py --data data/sift_query10000_k50.h5 \
  --output model/sift/lstm_io/k50 --epochs 30 --batch-size 32 \
  --workers 0 --seed 42 --device auto
```

Each output directory receives `latest.pt`, `best.pt`, `metrics.json`, and
`split_manifest.json`. The embedded 8,000 training queries are deterministically
split into 7,200 actual training queries and 800 validation queries. `best.pt`
is selected from epochs whose validation unique-prefetch recall is at least
0.95: first minimize `wasted_io_ratio`, then minimize
`tp_decision_delay_mean`. If no epoch reaches 0.95 recall, the highest-recall
epoch is selected. The separate 2,000-query test split is evaluated only after
that checkpoint has been selected. AUC is retained as a diagnostic metric.

Training uses the original I/O-aware weighted BCE at every valid timestep:

```text
label 1 weight = 1
label 0 weight = 1 + search_progress
```

Test evaluation also simulates irreversible, unique-node prefetch decisions at
threshold 0.5. It reports `wasted_io_ratio = FP/(TP+FN)` and the mean timestep
delay from each node's first result-queue appearance to its first prefetch.

ROC-AUC and PR-AUC use timestep-rank observations, so a node that remains in the
queue for several timesteps contributes several scores. AUC sweeps thresholds
over continuous scores; it is not calculated from only the TP/FP/TN/FN values
at threshold 0.5.
