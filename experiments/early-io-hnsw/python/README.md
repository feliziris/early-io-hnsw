# HNSW LSTM / GRU training

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

## GRU model selection

Use the same HDF5 traces, loss, and metrics with a two-layer GRU (hidden size 128):

```bash
python3 train.py \
  --data /path/to/hnsw_trace.h5 \
  --output /path/to/gru_io/k10 \
  --model-type gru --num-layers 2 --hidden-size 128

python3 inference.py --topk 10 \
  --data /path/to/hnsw_trace.h5 --checkpoint /path/to/gru_io/k10/best.pt \
  --raw-data /path/to/payload4k_10m.bin --payload-size 4096
```

`--model-type` defaults to `lstm`. Checkpoints store `HnswLstm` or `HnswGru`
in the `model` field; inference and ONNX export select the class automatically.
Older checkpoints without that field are treated as LSTM. GRU training requires
new weights; existing LSTM checkpoints cannot be reused as GRU weights.

## Stateful ONNX export

Export LSTM or GRU checkpoints as one-timestep streaming models:

```bash
python3 export_onnx.py \
  --topk 10 20 50
```

Use `--checkpoint-root /path/to/gru_io` to export GRU checkpoints from
`k10/best.pt`, `k20/best.pt`, and `k50/best.pt`. The data root must contain
`sift_query10000_k<K>.h5` validation files; specify it with `--data-root`.

Both architectures accept `features [1,k,6]` and `hidden_in [layers,1,hidden_size]`,
and return `probabilities [1,k]` and `hidden_out`. LSTM additionally accepts
`cell_in` and returns `cell_out`, with the same shape as the hidden state.
The Faiss ONNX runner detects this interface automatically; pass either model
using the existing `--onnx-model` option. Layer count and hidden size come from
the ONNX shapes. HDF5 is used for export validation, not Faiss runtime inference.
Feature normalization from `best.pt` is embedded in the graph. Models
and validation reports are written beside each checkpoint as `model.onnx` and
`onnx_export.json`. Export validation compares full-sequence PyTorch,
one-timestep PyTorch, and one-timestep ONNX Runtime outputs on a real test
trace.

### CPU/GPU streaming pipeline in Faiss

Build with `FAISS_ENABLE_ONNX_RUNTIME=ON` and
`FAISS_ENABLE_HNSW_CUDA_PIPELINE=ON` (requires CUDA Toolkit). Add these options
to `generate_sift_query_labels` along with the model and usual search/I/O options:

```bash
--onnx-model /path/to/model.onnx \
--onnx-provider cuda --onnx-inference async --onnx-gpu-inflight 8
```

The HNSW search thread copies feature snapshots into a per-query FIFO and
continues searching. The reusable CPU inference worker copies each snapshot
into an owned pinned-memory slot, enqueues host-to-device transfer and ONNX
execution, and collects completed predictions in timestep order. It checks the
threshold and submits deduplicated raw-data reads on the CPU. A single persistent
CUDA stream preserves hidden/cell dependencies between successive timesteps;
timesteps are not independent batches. The model/session/stream stay alive when
the GPU has no work. Recurrent state is initialized separately for each query.

`--onnx-gpu-inflight` bounds slots holding submitted predictions, not I/O queue
depth. A slot is reused only after its completion event and CPU result copy.
When the GPU lags, snapshots accumulate in the CPU FIFO; when it is fast, the
queues may stay empty. The worker waits for a GPU result only when slots are full
or search has ended; otherwise it polls completion while accepting new snapshots.
ORT asynchronous execution is requested, but host-side Run/launch overhead can
still limit overlap. No CUDA Graph optimization is enabled. When live payload I/O is enabled, traversal completion cancels queued snapshots
and immediately submits missing final-top-k reads. Already-running CPU inference
or submitted CUDA work is finalized concurrently with payload I/O; it is joined
before search returns to preserve object lifetimes. Trace-only runs without a
payload reader still drain all predictions for validation. Existing CPU/synchronous CUDA behavior is unchanged
when the new option is omitted. The benchmark wrapper forwards
`ONNX_GPU_INFLIGHT=8` with `ONNX_PROVIDER=cuda` and `ONNX_INFERENCE=async`.

`validate_hnsw_cuda_pipeline MODEL.onnx 200` checks CPU/synchronous CUDA versus
pipeline output, recurrent-state reset across queries, and bounded-slot reuse
at depths 1, 2, and 8. Optional arguments `INDEX QUERY.fvecs SIDECAR RAW`
also validate one real HNSW query and 4-KiB prefetch decisions. A peak pending
count alone does not establish concurrent GPU/CPU execution.

For a diagnostic GPU-pipeline profile of one query:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-onnx/demos/profile_hnsw_single_query \
  INDEX QUERY.fvecs SIDECAR MODEL.onnx RAW OUTPUT.json 0 cuda 8
python3 experiments/early-io-hnsw/python/plot_single_query_profile.py \
  OUTPUT.json --output workspace/results/figure/single_query_gpu.png
```

The executable uses k from the model, efSearch=200, threshold=0.5, four I/O
workers, and 4-KiB payloads. CPU-reference and actual-provider warmups run
outside the measured query. The JSON separates critical-path search phases,
snapshot FIFO delays, and inference-thread input/binding, Run call, transfer
enqueue, completion collection, and threshold/I/O work. These are CPU-side
wall-clock measurements: Run call duration is not a pure GPU-kernel timer.
Inference and search lanes overlap and must not be summed. Instrumentation
adds overhead; this single-query diagnostic is not a throughput benchmark.

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

### Stop prediction at traversal completion

Live asynchronous search no longer drains its snapshot backlog before issuing
final-result reads. It discards unstarted snapshots, closes speculative I/O at
final-result accounting, reuses previously submitted final-node requests, and
submits demand reads for the remaining nodes on the same worker queue. Reads
already submitted for false positives are not cancelled.

Unprocessed timesteps retain empty prediction/decision arrays in the trace.
The generator counts only completed predictions; replay recording emits empty
entries for cancelled timesteps. Traces without a payload reader remain complete.
The diagnostic profile reports `prediction_policy=stop_at_graph_end`,
`search_call_ms`, and `prediction_cleanup_after_data_ready_ms`. Payload-ready
end-to-end timing excludes subsequent inference/trace cleanup; the search call
includes it. `inference_end` can therefore follow `data_ready`. The legacy
`inference_drain_and_writeback` critical-path phase is zero for this policy.

Run the synthetic cancellation/concurrent-I/O regression validator with:

```bash
cmake --build build-onnx-cpu --target validate_hnsw_async_stop
build-onnx-cpu/demos/validate_hnsw_async_stop MODEL.onnx
```
