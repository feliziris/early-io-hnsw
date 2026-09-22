# HNSW LSTM training

This directory contains the new offline trainer. The previous experimental
Python implementation is preserved under `tmp/`.

The trainer uses exactly six features, in this order:

1. `ratio_to_entry`
2. `ratio_to_first`
3. `search_progress`
4. `rank_norm`
5. `persistence_rate`
6. `avg_neigh_dist`

## Input HDF5 schema

The root attribute `feature_names` should be a JSON string containing the six
names above. Each query is stored as a group (for example `query_00000`) with:

```text
query_00000/features          float32 [T, k, 6]
query_00000/labels            uint8   [T, k]
query_00000/node_ids          int32   [T, k]
query_00000/query_distances   float32 [T, k]
query_00000/final_ids         int32   [k]
```

`labels[t, r]` is 1 when the node at rank `r` at timestep `t` belongs to the
query's final top-k, and 0 otherwise. All query groups in one file must use the
same `k`. The final timestep is excluded from training because its queue already
equals the final result and therefore supplies a trivial target.

Feature normalization is fitted using training queries only. The resulting mean
and standard deviation are stored in `best.pt` for use during Faiss runtime
inference.

## Training

```bash
python3 train.py \
  --data /path/to/hnsw_trace.h5 \
  --output /path/to/checkpoint_directory
```

The converter embeds a seed-42 query-level split containing exactly 80% train
queries and 20% test queries. Training further reserves 10% of the embedded
train split for validation, producing 7,200 train / 800 validation / 2,000 test
queries. `best.pt` requires validation unique-prefetch recall >= 0.95, then
minimizes wasted I/O and TP decision delay in that order. If no epoch meets the
recall constraint, the highest-recall epoch is used. The default model is a
two-layer stacked LSTM with hidden size 128. One sample is one complete query
trace. Loss is applied to all
non-final timesteps in one causal LSTM pass; the final, already-settled result
queue is excluded. This avoids repeated computation of overlapping prefixes.
The loss is the original `io_aware` weighted BCE: positive labels have weight
1, while negative labels have weight `1 + search_progress`.

The complete reproducible generation, conversion and training pipeline is:

```bash
bash run_pipeline.sh
```

The Faiss runtime inference program will be kept separate from this offline
trainer. It must use the feature order and normalization values saved in
`best.pt`.

## Offline inference

Use the same `inference.py` for every trained top-k model. `--topk` selects
both the default HDF5 file and checkpoint, and `--threshold` controls the
prefetch decision threshold:

```bash
python3 inference.py --topk 10 --threshold 0.5
python3 inference.py --topk 20 --threshold 0.5
python3 inference.py --topk 50 --threshold 0.5
```

Each first threshold crossing submits that node's 512 KiB payload to an
asynchronous mmap reader queue. Model inference continues while the reader
threads perform the individual payload reads. The default raw file is
`workspace/raw_data/payload512k.bin`; change the I/O setup with
`--raw-data`, `--payload-size`, `--io-workers`, and `--io-queue-size`:

```bash
python3 inference.py \
  --topk 10 \
  --prefetch on \
  --evict-file-cache \
  --threshold 0.5 \
  --io-workers 4 \
  --io-queue-size 4096
```

Run the no-prefetch baseline with the same model and test split:

```bash
python3 inference.py \
  --topk 10 \
  --prefetch off \
  --evict-file-cache \
  --threshold 0.5
```

`--evict-file-cache` calls `POSIX_FADV_DONTNEED` on the raw file before it is
memory-mapped. This is a per-file kernel eviction hint used to reduce page-
cache carryover between benchmark runs without globally dropping the machine's
caches or requiring root privileges.

In baseline mode, no predicted node is submitted to the asynchronous queue.
When each query's final top-k becomes available, all final payloads are read
synchronously. With prefetch enabled, predicted payloads are submitted to the
asynchronous queue; at query completion the program waits for prefetched final
nodes and synchronously reads any final nodes that were missed. Both modes
therefore measure the same final-top-k data-ready boundary.

The process drains all submitted I/O before writing metrics. The output also
contains mmap read latency, queue-wait latency, and full decision-to-I/O-
completion latency distributions.

The defaults resolve to `output/sift_query10000_k<TOPK>.h5` and
`workspace/model/sift/lstm_io/k<TOPK>/best.pt` relative to the repository
root. Use `--data` or `--checkpoint` only to override those paths. Results are
written beside the selected checkpoint as `inference_metrics_prefetch_on.json`
or `inference_metrics_prefetch_off.json` unless `--output` is specified. The
JSON contains `tp_decision_delay`, its sum and count, and `wasted_io_ratio`
with its numerator and denominator. The `baseline_latency` or
`prefetch_latency` section reports total elapsed time and query data-ready wait
latency. These are offline trace-replay measurements and do not include a live
Faiss HNSW search.

Test metrics include ordinary timestep/rank classification counts and a
separate unique-node prefetch simulation. A node is prefetched only the first
time its score reaches `--prefetch-threshold` (default 0.5); later appearances
are ignored. `wasted_io_ratio` is `prefetch_fp / (prefetch_tp + prefetch_fn)`.
`decision_delay_mean` is the mean difference between a node's first result-queue
appearance timestep and its first prefetch-decision timestep.
