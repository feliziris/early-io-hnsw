#!/usr/bin/env python3
"""Run an HNSW LSTM checkpoint on the embedded HDF5 test split.

Each node can trigger at most one prefetch decision per query.  Metrics are
written as JSON so that the exact counts behind the two requested aggregate
metrics remain available.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HnswSequenceDataset, collate_sequences, read_split
from model import HnswLstm
from schema import FEATURE_COUNT, FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HNSW LSTM on the HDF5 test split."
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=True,
        choices=(10, 20, 50),
        help="top-k model and dataset to evaluate",
    )
    parser.add_argument(
        "--data",
        help="override the default output/sift_query10000_k<TOPK>.h5",
    )
    parser.add_argument(
        "--checkpoint",
        help="override the default model/sift/lstm_io/k<TOPK>/best.pt",
    )
    parser.add_argument(
        "--output",
        help=(
            "metrics JSON (default: <checkpoint directory>/"
            "inference_metrics_prefetch_<on|off>.json)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--raw-data",
        help="raw payload file (default: raw_data/payload512k.bin)",
    )
    parser.add_argument(
        "--evict-file-cache",
        action="store_true",
        help=(
            "advise the kernel to evict cached pages for the raw file before "
            "mmap (Linux POSIX_FADV_DONTNEED)"
        ),
    )
    parser.add_argument(
        "--prefetch",
        choices=("on", "off"),
        default="on",
        help=(
            "on: asynchronously read predicted nodes; off: baseline that reads "
            "final top-k payloads synchronously (default: on)"
        ),
    )
    parser.add_argument(
        "--payload-size",
        type=int,
        default=512 * 1024,
        help="bytes assigned to each node (default: 524288)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
        help="number of asynchronous mmap reader threads (default: 4)",
    )
    parser.add_argument(
        "--io-queue-size",
        type=int,
        default=4096,
        help="maximum queued prefetch requests before backpressure (default: 4096)",
    )
    parser.add_argument(
        "--threshold",
        "--prefetch-threshold",
        dest="threshold",
        type=float,
        default=None,
        help=(
            "prefetch decision threshold; if omitted, use the value stored "
            "in the checkpoint"
        ),
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")
    return device


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_config",
        "state_dict",
        "feature_names",
        "feature_mean",
        "feature_std",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")

    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError(
            f"checkpoint feature order {checkpoint['feature_names']} does not match "
            f"the expected order {FEATURE_NAMES}"
        )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    if mean.shape != (FEATURE_COUNT,) or std.shape != (FEATURE_COUNT,):
        raise ValueError("checkpoint feature_mean and feature_std must each have 6 values")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("checkpoint normalization contains invalid values")

    config = checkpoint["model_config"]
    model = HnswLstm(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model, mean, std


class RawReadRequest:
    def __init__(self, node_id: int, decision_ns: int) -> None:
        self.node_id = node_id
        self.decision_ns = decision_ns
        self.done = threading.Event()
        self.error: BaseException | None = None

    def wait(self) -> None:
        self.done.wait()
        if self.error is not None:
            raise RuntimeError(f"asynchronous read failed for node {self.node_id}") from self.error


class AsyncRawPrefetcher:
    """Read one fixed-size mmap payload per decision on background threads."""

    _STOP = object()

    def __init__(
        self,
        path: Path,
        payload_size: int,
        worker_count: int,
        queue_size: int,
        evict_file_cache: bool = False,
    ) -> None:
        self.path = path
        self.payload_size = payload_size
        self.worker_count = worker_count
        self._file = path.open("rb", buffering=0)
        self.file_size = os.fstat(self._file.fileno()).st_size
        if self.file_size == 0:
            self._file.close()
            raise ValueError(f"raw data file is empty: {path}")
        if self.file_size % payload_size != 0:
            self._file.close()
            raise ValueError(
                f"raw data size {self.file_size} is not divisible by payload size "
                f"{payload_size}"
            )
        self.cache_eviction_advised = evict_file_cache
        if evict_file_cache:
            if not hasattr(os, "posix_fadvise") or not hasattr(
                os, "POSIX_FADV_DONTNEED"
            ):
                self._file.close()
                raise RuntimeError(
                    "--evict-file-cache is not supported by this platform"
                )
            os.posix_fadvise(
                self._file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
            )
        self.payload_count = self.file_size // payload_size
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._queue_wait_ns: list[int] = []
        self._read_ns: list[int] = []
        self._completion_ns: list[int] = []
        self._demand_read_ns: list[int] = []
        self._query_data_wait_ns: list[int] = []
        self._errors: list[BaseException] = []
        self._closed = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"raw-prefetch-{worker_index}",
                daemon=True,
            )
            for worker_index in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def _validate_node_id(self, node_id: int) -> None:
        if not 0 <= node_id < self.payload_count:
            raise IndexError(
                f"node ID {node_id} is outside raw payload range "
                f"[0, {self.payload_count})"
            )

    def submit(self, node_id: int) -> RawReadRequest:
        if self._closed:
            raise RuntimeError("cannot submit to a closed raw prefetcher")
        self._validate_node_id(node_id)
        # Record the model's decision time before queueing. If the bounded
        # queue fills, the resulting backpressure is included in completion
        # latency instead of silently dropping a prefetch.
        request = RawReadRequest(node_id, time.perf_counter_ns())
        self._queue.put(request)
        return request

    def _read_payload(self, node_id: int) -> None:
        offset = node_id * self.payload_size
        payload = self._mapping[offset : offset + self.payload_size]
        if len(payload) != self.payload_size:
            raise IOError(
                f"short mmap read for node {node_id}: "
                f"{len(payload)} != {self.payload_size}"
            )

    def read_demand(self, node_id: int) -> None:
        """Synchronously read a final-result payload that was not prefetched."""
        self._validate_node_id(node_id)
        started_ns = time.perf_counter_ns()
        self._read_payload(node_id)
        finished_ns = time.perf_counter_ns()
        with self._lock:
            self._demand_read_ns.append(finished_ns - started_ns)

    def record_query_data_wait(self, elapsed_ns: int) -> None:
        with self._lock:
            self._query_data_wait_ns.append(elapsed_ns)

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._STOP:
                    return
                request = task
                node_id = request.node_id
                started_ns = time.perf_counter_ns()
                # A slice creates a bytes object, forcing all 512 KiB to be
                # read from the mapping rather than touching only one page.
                self._read_payload(node_id)
                finished_ns = time.perf_counter_ns()
                with self._lock:
                    self._queue_wait_ns.append(started_ns - request.decision_ns)
                    self._read_ns.append(finished_ns - started_ns)
                    self._completion_ns.append(finished_ns - request.decision_ns)
            except BaseException as error:
                if task is not self._STOP:
                    task.error = error
                with self._lock:
                    self._errors.append(error)
            finally:
                if task is not self._STOP:
                    task.done.set()
                self._queue.task_done()

    @staticmethod
    def _latency_summary(values_ns: list[int]) -> dict[str, float | int]:
        if not values_ns:
            return {
                "count": 0,
                "mean_us": 0.0,
                "min_us": 0.0,
                "p50_us": 0.0,
                "p95_us": 0.0,
                "p99_us": 0.0,
                "max_us": 0.0,
            }
        values_us = np.asarray(values_ns, dtype=np.float64) / 1000.0
        p50, p95, p99 = np.percentile(values_us, [50, 95, 99])
        return {
            "count": len(values_ns),
            "mean_us": float(values_us.mean()),
            "min_us": float(values_us.min()),
            "p50_us": float(p50),
            "p95_us": float(p95),
            "p99_us": float(p99),
            "max_us": float(values_us.max()),
        }

    def finish(self) -> dict:
        """Drain every submitted request, stop workers, and return I/O metrics."""
        if self._closed:
            raise RuntimeError("raw prefetcher was already closed")
        self._closed = True
        for _ in self._threads:
            self._queue.put(self._STOP)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        self._mapping.close()
        self._file.close()
        if self._errors:
            raise RuntimeError(
                f"{len(self._errors)} asynchronous raw read(s) failed"
            ) from self._errors[0]

        async_read_count = len(self._read_ns)
        demand_read_count = len(self._demand_read_ns)
        read_count = async_read_count + demand_read_count
        return {
            "mode": "asynchronous_mmap",
            "raw_data": str(self.path),
            "payload_size_bytes": self.payload_size,
            "available_payload_count": self.payload_count,
            "cache_eviction_advised_before_mmap": self.cache_eviction_advised,
            "io_workers": self.worker_count,
            "read_count": read_count,
            "total_bytes_read": read_count * self.payload_size,
            "async_prefetch_read_count": async_read_count,
            "synchronous_demand_read_count": demand_read_count,
            "queue_wait_latency": self._latency_summary(self._queue_wait_ns),
            "mmap_read_latency": self._latency_summary(self._read_ns),
            "synchronous_demand_read_latency": self._latency_summary(
                self._demand_read_ns
            ),
            "decision_to_completion_latency": self._latency_summary(
                self._completion_ns
            ),
            "query_data_ready_wait_latency": self._latency_summary(
                self._query_data_wait_ns
            ),
        }


def new_accumulator() -> dict[str, int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tp_delay_sum": 0,
        "tp_delay_count": 0,
    }


def update_metrics(
    accumulator: dict[str, int],
    probabilities: torch.Tensor,
    node_ids: torch.Tensor,
    final_ids: torch.Tensor,
    lengths: torch.Tensor,
    threshold: float,
    prefetcher: AsyncRawPrefetcher,
    prefetch_enabled: bool,
) -> None:
    """Apply the first-threshold-crossing prefetch rule independently per query."""
    for batch_index in range(probabilities.shape[0]):
        timestep_count = int(lengths[batch_index])
        final_list = [
            int(node_id)
            for node_id in final_ids[batch_index].tolist()
            if int(node_id) >= 0
        ]
        final_set = set(final_list)
        first_entry: dict[int, int] = {}
        prefetched: set[int] = set()
        requests: dict[int, RawReadRequest] = {}

        for timestep in range(timestep_count):
            for rank in range(node_ids.shape[2]):
                node_id = int(node_ids[batch_index, timestep, rank])
                if node_id < 0:
                    continue
                first_entry.setdefault(node_id, timestep)

                # Once prefetched, a node is never predicted or counted again.
                if node_id in prefetched:
                    continue
                if float(probabilities[batch_index, timestep, rank]) < threshold:
                    continue

                prefetched.add(node_id)
                if prefetch_enabled:
                    requests[node_id] = prefetcher.submit(node_id)
                if node_id in final_set:
                    accumulator["tp_delay_sum"] += timestep - first_entry[node_id]
                    accumulator["tp_delay_count"] += 1

        # Data-ready latency starts when the final result becomes available.
        # With prefetch enabled, wait for in-flight true positives and read
        # false negatives synchronously. The baseline reads every final node
        # synchronously in rank order.
        data_wait_started_ns = time.perf_counter_ns()
        for node_id in final_list:
            request = requests.get(node_id) if prefetch_enabled else None
            if request is not None:
                request.wait()
            else:
                prefetcher.read_demand(node_id)
        prefetcher.record_query_data_wait(
            time.perf_counter_ns() - data_wait_started_ns
        )

        accumulator["tp"] += len(prefetched & final_set)
        accumulator["fp"] += len(prefetched - final_set)
        accumulator["fn"] += len(final_set - prefetched)


def infer(
    model: HnswLstm,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    prefetcher: AsyncRawPrefetcher,
    prefetch_enabled: bool,
) -> dict[str, int]:
    accumulator = new_accumulator()
    with torch.inference_mode():
        for batch_index, (features, _, _, node_ids, final_ids, lengths) in enumerate(
            loader, start=1
        ):
            logits = model(features.to(device, non_blocking=True), lengths)
            update_metrics(
                accumulator,
                logits.sigmoid().cpu(),
                node_ids,
                final_ids,
                lengths,
                threshold,
                prefetcher,
                prefetch_enabled,
            )
            if batch_index % 10 == 0 or batch_index == len(loader):
                completed = min(batch_index * loader.batch_size, len(loader.dataset))
                print(f"test queries: {completed}/{len(loader.dataset)}", flush=True)
    return accumulator


def write_json_atomic(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.payload_size <= 0:
        raise ValueError("payload-size must be positive")
    if args.io_workers <= 0:
        raise ValueError("io-workers must be positive")
    if args.io_queue_size <= 0:
        raise ValueError("io-queue-size must be positive")

    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        Path(args.data).resolve()
        if args.data
        else project_root / "output" / f"sift_query10000_k{args.topk}.h5"
    )
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else project_root
        / "model"
        / "sift"
        / "lstm_io"
        / f"k{args.topk}"
        / "best.pt"
    )
    raw_data_path = (
        Path(args.raw_data).resolve()
        if args.raw_data
        else project_root / "raw_data" / "payload512k.bin"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else checkpoint_path.parent / f"inference_metrics_prefetch_{args.prefetch}.json"
    )
    device = select_device(args.device)
    checkpoint, model, mean, std = load_checkpoint(checkpoint_path, device)

    _, test_ids = read_split(str(data_path))
    dataset = HnswSequenceDataset(str(data_path), test_ids, mean, std)
    model_config = checkpoint["model_config"]
    if dataset.k != args.topk:
        raise ValueError(f"--topk={args.topk} does not match data k={dataset.k}")
    if dataset.k != int(model_config["k"]):
        raise ValueError(
            f"data k={dataset.k} does not match checkpoint k={model_config['k']}"
        )
    if dataset.input_size != int(model_config["input_size"]):
        raise ValueError(
            f"data input_size={dataset.input_size} does not match checkpoint "
            f"input_size={model_config['input_size']}"
        )

    stored_threshold = checkpoint.get("metadata", {}).get("prefetch_threshold", 0.5)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(stored_threshold)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("prefetch-threshold must be between 0 and 1")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_sequences,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    print(
        f"device={device} k={dataset.k} test_queries={len(dataset)} "
        f"prefetch={args.prefetch} prefetch_threshold={threshold} "
        f"io_workers={args.io_workers} "
        f"raw_data={raw_data_path}",
        flush=True,
    )
    prefetch_enabled = args.prefetch == "on"
    started_at = time.perf_counter()
    prefetcher = AsyncRawPrefetcher(
        raw_data_path,
        payload_size=args.payload_size,
        worker_count=args.io_workers,
        queue_size=args.io_queue_size,
        evict_file_cache=args.evict_file_cache,
    )
    try:
        accumulator = infer(
            model, loader, device, threshold, prefetcher, prefetch_enabled
        )
        inference_seconds = time.perf_counter() - started_at
    finally:
        raw_io = prefetcher.finish()
    raw_io["mode"] = (
        "asynchronous_prefetch_plus_synchronous_demand"
        if prefetch_enabled
        else "synchronous_demand_baseline"
    )
    elapsed_seconds = time.perf_counter() - started_at

    tp = accumulator["tp"]
    fp = accumulator["fp"]
    fn = accumulator["fn"]
    final_positive_count = tp + fn
    expected_async_reads = tp + fp if prefetch_enabled else 0
    expected_demand_reads = fn if prefetch_enabled else final_positive_count
    if raw_io["async_prefetch_read_count"] != expected_async_reads:
        raise RuntimeError(
            f"async read count {raw_io['async_prefetch_read_count']} does not match "
            f"expected count {expected_async_reads}"
        )
    if raw_io["synchronous_demand_read_count"] != expected_demand_reads:
        raise RuntimeError(
            f"demand read count {raw_io['synchronous_demand_read_count']} does not "
            f"match expected count {expected_demand_reads}"
        )
    delay_count = accumulator["tp_delay_count"]
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "data": str(data_path),
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "inference_and_submission_seconds": inference_seconds,
        "feature_names": list(FEATURE_NAMES),
        "k": dataset.k,
        "prefetch": args.prefetch,
        "prefetch_threshold": threshold,
        "test_query_count": len(dataset),
        "metrics": {
            "tp_decision_delay": (
                accumulator["tp_delay_sum"] / delay_count if delay_count else 0.0
            ),
            "tp_decision_delay_sum": accumulator["tp_delay_sum"],
            "tp_decision_delay_count": delay_count,
            "wasted_io_ratio": fp / final_positive_count if final_positive_count else 0.0,
            "wasted_io_numerator_fp": fp,
            "wasted_io_denominator_tp_plus_fn": final_positive_count,
            "prefetch_tp": tp,
            "prefetch_fp": fp,
            "prefetch_fn": fn,
            "prefetch_recall": tp / final_positive_count if final_positive_count else 0.0,
        },
        "raw_io": raw_io,
    }
    performance_key = "prefetch_latency" if prefetch_enabled else "baseline_latency"
    result[performance_key] = {
        "scope": (
            "offline inference plus final-top-k data-ready wait; "
            "does not include a live Faiss HNSW search"
        ),
        "elapsed_seconds": elapsed_seconds,
        "query_data_ready_wait_latency": raw_io[
            "query_data_ready_wait_latency"
        ],
    }
    write_json_atomic(output_path, result)
    print(
        f"tp_decision_delay={result['metrics']['tp_decision_delay']:.6f} "
        f"wasted_io_ratio={result['metrics']['wasted_io_ratio']:.6f} "
        f"raw_reads={raw_io['read_count']}",
        flush=True,
    )
    print(f"metrics written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
