#!/usr/bin/env python3
"""Convert an HNSWTRC1 stream from the Faiss demo into validated HDF5."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import h5py
import numpy as np

from schema import FEATURE_COUNT, FEATURE_NAMES, query_group_name


MAGIC = b"HNSWTRC1"
HEADER = struct.Struct("<8s6I3Q")
QUERY_HEADER = struct.Struct("<IIf")
TIMESTEP_HEADER = struct.Struct("<Qiff")


def read_array(handle, dtype: str, count: int, context: str) -> np.ndarray:
    result = np.fromfile(handle, dtype=np.dtype(dtype), count=count)
    if result.size != count:
        raise EOFError(f"truncated trace while reading {context}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be between 0 and 1")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    observed_timesteps = observed_labels = observed_positives = 0
    try:
        with open(args.trace, "rb") as source:
            raw_header = source.read(HEADER.size)
            if len(raw_header) != HEADER.size:
                raise EOFError("truncated trace header")
            unpacked = HEADER.unpack(raw_header)
            magic, version, query_count, k, ef_search, feature_count, reserved = unpacked[:7]
            expected_counts = unpacked[7:]
            if magic != MAGIC or version != 1:
                raise ValueError(f"unsupported trace format: magic={magic!r}, version={version}")
            if feature_count != FEATURE_COUNT or reserved != 0:
                raise ValueError("trace feature metadata is invalid")

            rng = np.random.default_rng(args.seed)
            permutation = rng.permutation(query_count).astype(np.int32)
            train_count = int(query_count * args.train_ratio)
            train_ids, test_ids = permutation[:train_count], permutation[train_count:]

            with h5py.File(temporary, "w") as target:
                target.attrs.update({
                    "format": "HNSWTRC1-HDF5", "format_version": 1,
                    "query_count": query_count, "k": k, "ef_search": ef_search,
                    "feature_count": feature_count,
                    "feature_names": json.dumps(FEATURE_NAMES),
                    "split_seed": args.seed, "train_ratio": args.train_ratio,
                })
                target.create_dataset("train_query_ids", data=train_ids)
                target.create_dataset("test_query_ids", data=test_ids)
                seen_ids = set()
                for record_number in range(query_count):
                    raw = source.read(QUERY_HEADER.size)
                    if len(raw) != QUERY_HEADER.size:
                        raise EOFError(f"truncated query header at record {record_number}")
                    query_id, timestep_count, entry_distance = QUERY_HEADER.unpack(raw)
                    if query_id in seen_ids or query_id >= query_count:
                        raise ValueError(f"invalid or duplicate query ID {query_id}")
                    seen_ids.add(query_id)
                    final_ids = read_array(source, "<i4", k, "final IDs")
                    final_distances = read_array(source, "<f4", k, "final distances")
                    pop_count = np.empty(timestep_count, dtype=np.uint64)
                    expanded_ids = np.empty(timestep_count, dtype=np.int32)
                    first_distance = np.empty(timestep_count, dtype=np.float32)
                    search_progress = np.empty(timestep_count, dtype=np.float32)
                    node_ids = np.empty((timestep_count, k), dtype=np.int32)
                    query_distances = np.empty((timestep_count, k), dtype=np.float32)
                    features = np.empty((timestep_count, k, FEATURE_COUNT), dtype=np.float32)
                    labels = np.empty((timestep_count, k), dtype=np.uint8)
                    for timestep in range(timestep_count):
                        raw = source.read(TIMESTEP_HEADER.size)
                        if len(raw) != TIMESTEP_HEADER.size:
                            raise EOFError(f"query {query_id}: truncated timestep {timestep}")
                        values = TIMESTEP_HEADER.unpack(raw)
                        pop_count[timestep], expanded_ids[timestep] = values[:2]
                        first_distance[timestep], search_progress[timestep] = values[2:]
                        node_ids[timestep] = read_array(source, "<i4", k, "node IDs")
                        query_distances[timestep] = read_array(source, "<f4", k, "query distances")
                        features[timestep] = read_array(
                            source, "<f4", k * FEATURE_COUNT, "features"
                        ).reshape(k, FEATURE_COUNT)
                        labels[timestep] = read_array(source, "u1", k, "labels")
                    expected = np.isin(node_ids, final_ids).astype(np.uint8)
                    if not np.array_equal(labels, expected):
                        raise ValueError(f"query {query_id}: label validation failed")
                    if not np.isfinite(features).all() or not np.isfinite(query_distances).all():
                        raise ValueError(f"query {query_id}: non-finite feature or distance")
                    if not np.allclose(features[:, :, 2], search_progress[:, None]):
                        raise ValueError(f"query {query_id}: search_progress feature mismatch")
                    group = target.create_group(query_group_name(query_id))
                    group.attrs.update({"query_id": query_id, "entry_distance": entry_distance,
                                        "timestep_count": timestep_count})
                    options = {"compression": "lzf", "shuffle": True}
                    for name, data in (
                        ("features", features), ("labels", labels), ("node_ids", node_ids),
                        ("query_distances", query_distances), ("pop_count", pop_count),
                        ("expanded_node_id", expanded_ids), ("first_distance", first_distance),
                        ("search_progress", search_progress),
                    ):
                        group.create_dataset(name, data=data, **options)
                    group.create_dataset("final_ids", data=final_ids)
                    group.create_dataset("final_distances", data=final_distances)
                    observed_timesteps += timestep_count
                    observed_labels += labels.size
                    observed_positives += int(labels.sum())
                    if (record_number + 1) % 1000 == 0 or record_number + 1 == query_count:
                        print(f"converted_queries={record_number + 1}/{query_count}", flush=True)
                if source.read(1):
                    raise ValueError("unexpected trailing bytes in trace")
                if seen_ids != set(range(query_count)):
                    raise ValueError("trace must contain query IDs 0..nq-1")
                observed = (observed_timesteps, observed_labels, observed_positives)
                if observed != expected_counts:
                    raise ValueError(f"header totals mismatch: expected={expected_counts}, observed={observed}")
                target.attrs["total_timesteps"] = observed_timesteps
                target.attrs["total_labels"] = observed_labels
                target.attrs["positive_labels"] = observed_positives
        os.replace(temporary, output)
        print(f"output={output}")
        print(f"train_queries={train_count} test_queries={query_count - train_count}")
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


if __name__ == "__main__":
    main()
