#!/usr/bin/env python3
"""Read and validate files produced by generate_sift_query_labels."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


HEADER = struct.Struct("<8s6I3Q")
QUERY_HEADER = struct.Struct("<IIf")
TIMESTEP_HEADER = struct.Struct("<Qif")
NODE_LABEL = struct.Struct("<iB")
MAGIC = b"HNSWLBL1"


@dataclass(frozen=True)
class Header:
    version: int
    nq: int
    k: int
    ef_search: int
    feature_count: int
    total_timesteps: int
    total_labels: int
    positive_labels: int


@dataclass(frozen=True)
class Timestep:
    pop_count: int
    expanded_node_id: int
    search_progress: float
    node_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class QueryLabels:
    query_id: int
    entry_distance: float
    final_ids: tuple[int, ...]
    final_distances: tuple[float, ...]
    timesteps: tuple[Timestep, ...]


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"unexpected EOF: requested {size}, got {len(data)}")
    return data


def read_header(stream: BinaryIO) -> Header:
    (
        magic,
        version,
        nq,
        k,
        ef_search,
        feature_count,
        _reserved,
        total_timesteps,
        total_labels,
        positive_labels,
    ) = HEADER.unpack(read_exact(stream, HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported version: {version}")
    return Header(
        version,
        nq,
        k,
        ef_search,
        feature_count,
        total_timesteps,
        total_labels,
        positive_labels,
    )


def iter_queries(stream: BinaryIO, header: Header) -> Iterator[QueryLabels]:
    ids_struct = struct.Struct(f"<{header.k}i")
    distances_struct = struct.Struct(f"<{header.k}f")
    for expected_query_id in range(header.nq):
        query_id, timestep_count, entry_distance = QUERY_HEADER.unpack(
            read_exact(stream, QUERY_HEADER.size)
        )
        if query_id != expected_query_id:
            raise ValueError(
                f"query ID mismatch: expected {expected_query_id}, got {query_id}"
            )
        final_ids = ids_struct.unpack(read_exact(stream, ids_struct.size))
        final_distances = distances_struct.unpack(
            read_exact(stream, distances_struct.size)
        )
        timesteps: list[Timestep] = []
        final_set = set(final_ids)
        for _ in range(timestep_count):
            pop_count, expanded_node_id, progress = TIMESTEP_HEADER.unpack(
                read_exact(stream, TIMESTEP_HEADER.size)
            )
            node_ids: list[int] = []
            labels: list[int] = []
            for _rank in range(header.k):
                node_id, label = NODE_LABEL.unpack(
                    read_exact(stream, NODE_LABEL.size)
                )
                if label not in (0, 1):
                    raise ValueError(f"invalid label value: {label}")
                expected_label = int(node_id in final_set)
                if label != expected_label:
                    raise ValueError(
                        f"label mismatch for query {query_id}, node {node_id}"
                    )
                node_ids.append(node_id)
                labels.append(label)
            timesteps.append(
                Timestep(
                    pop_count,
                    expanded_node_id,
                    progress,
                    tuple(node_ids),
                    tuple(labels),
                )
            )
        yield QueryLabels(
            query_id,
            entry_distance,
            final_ids,
            final_distances,
            tuple(timesteps),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--show-first", action="store_true")
    args = parser.parse_args()

    counted_queries = 0
    counted_timesteps = 0
    counted_labels = 0
    counted_positives = 0
    first_query: QueryLabels | None = None

    with args.labels.open("rb") as stream:
        header = read_header(stream)
        for query in iter_queries(stream, header):
            if first_query is None:
                first_query = query
            counted_queries += 1
            counted_timesteps += len(query.timesteps)
            counted_labels += len(query.timesteps) * header.k
            counted_positives += sum(
                sum(timestep.labels) for timestep in query.timesteps
            )
        if stream.read(1):
            raise ValueError("trailing bytes after final query record")

    expected = (
        header.nq,
        header.total_timesteps,
        header.total_labels,
        header.positive_labels,
    )
    counted = (
        counted_queries,
        counted_timesteps,
        counted_labels,
        counted_positives,
    )
    if counted != expected:
        raise ValueError(f"header totals {expected} != counted totals {counted}")

    print(f"queries={header.nq}")
    print(f"k={header.k}")
    print(f"ef_search={header.ef_search}")
    print(f"timesteps={header.total_timesteps}")
    print(f"labels={header.total_labels}")
    print(f"positive_labels={header.positive_labels}")
    print(f"negative_labels={header.total_labels - header.positive_labels}")
    print("validation=ok")

    if args.show_first and first_query and first_query.timesteps:
        timestep = first_query.timesteps[0]
        print(f"query_0_final_ids={list(first_query.final_ids)}")
        print(f"query_0_timestep_0_node_ids={list(timestep.node_ids)}")
        print(f"query_0_timestep_0_labels={list(timestep.labels)}")


if __name__ == "__main__":
    main()
