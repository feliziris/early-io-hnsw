#!/usr/bin/env python3
"""Extract the HDF5-embedded test split from an fvecs query file."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.hdf5, "r") as handle:
        if "test_query_ids" not in handle:
            raise ValueError("HDF5 file has no test_query_ids dataset")
        test_ids = np.asarray(handle["test_query_ids"], dtype=np.int64)
        split_seed = int(handle.attrs["split_seed"])
        train_ratio = float(handle.attrs["train_ratio"])

    if test_ids.ndim != 1 or test_ids.size == 0:
        raise ValueError("test_query_ids must be a non-empty vector")
    if np.unique(test_ids).size != test_ids.size or np.any(test_ids < 0):
        raise ValueError("test_query_ids contains invalid or duplicate IDs")

    source_size = args.source.stat().st_size
    with args.source.open("rb") as source:
        dimension_raw = source.read(4)
        if len(dimension_raw) != 4:
            raise ValueError("source fvecs is empty")
        dimension = struct.unpack("<i", dimension_raw)[0]
        if dimension <= 0:
            raise ValueError("source fvecs has an invalid dimension")
        record_size = 4 + dimension * 4
        if source_size % record_size != 0:
            raise ValueError("source size is not a multiple of its record size")
        source_count = source_size // record_size
        if int(test_ids.max()) >= source_count:
            raise ValueError("test query ID exceeds the source query count")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.mapping.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output, args.mapping.open(
            "w", encoding="utf-8"
        ) as mapping:
            mapping.write("test_row\toriginal_query_id\n")
            for test_row, query_id in enumerate(test_ids.tolist()):
                source.seek(query_id * record_size)
                record = source.read(record_size)
                if len(record) != record_size:
                    raise EOFError(f"truncated fvecs record for query {query_id}")
                record_dimension = struct.unpack_from("<i", record)[0]
                if record_dimension != dimension:
                    raise ValueError(
                        f"query {query_id} dimension {record_dimension} != {dimension}"
                    )
                output.write(record)
                mapping.write(f"{test_row}\t{query_id}\n")

    print(f"source_queries={source_count}")
    print(f"test_queries={test_ids.size}")
    print(f"dimension={dimension}")
    print(f"split_seed={split_seed}")
    print(f"train_ratio={train_ratio}")
    print(f"output={args.output}")
    print(f"mapping={args.mapping}")


if __name__ == "__main__":
    main()
