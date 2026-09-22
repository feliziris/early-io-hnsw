"""HDF5 dataset for six-feature HNSW search traces."""

from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from schema import FEATURE_COUNT, FEATURE_NAMES, query_group_name


def read_split(path: str) -> tuple[list[int], list[int]]:
    """Read the deterministic 80:20 query split embedded by the converter."""
    with h5py.File(path, "r") as handle:
        if "train_query_ids" not in handle or "test_query_ids" not in handle:
            raise ValueError("HDF5 file has no embedded train/test split")
        train_ids = handle["train_query_ids"][:].astype(np.int64).tolist()
        test_ids = handle["test_query_ids"][:].astype(np.int64).tolist()
    if not train_ids or not test_ids:
        raise ValueError("train and test splits must both be non-empty")
    if set(train_ids) & set(test_ids):
        raise ValueError("train/test query leakage detected")
    return train_ids, test_ids


def validate_trace_file(path: str) -> tuple[int, int]:
    """Validate global metadata and return (k, query_count)."""
    with h5py.File(path, "r") as handle:
        stored = handle.attrs.get("feature_names")
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        if stored is None or tuple(json.loads(str(stored))) != FEATURE_NAMES:
            raise ValueError(f"feature order must be {FEATURE_NAMES}")
        if int(handle.attrs.get("feature_count", -1)) != FEATURE_COUNT:
            raise ValueError("feature_count must be 6")
        k = int(handle.attrs["k"])
        query_count = int(handle.attrs["query_count"])
        train_ids = handle["train_query_ids"][:]
        test_ids = handle["test_query_ids"][:]
        if len(train_ids) + len(test_ids) != query_count:
            raise ValueError("embedded split does not cover every query")
    return k, query_count


def fit_normalization(path: str, query_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature normalization using training queries only."""
    total = np.zeros(FEATURE_COUNT, dtype=np.float64)
    square_total = np.zeros(FEATURE_COUNT, dtype=np.float64)
    count = 0
    with h5py.File(path, "r") as handle:
        for query_id in query_ids:
            values = np.asarray(
                handle[query_group_name(query_id)]["features"], dtype=np.float32
            ).reshape(-1, FEATURE_COUNT)
            if not np.isfinite(values).all():
                raise ValueError(f"query {query_id}: features contain NaN or infinity")
            total += values.sum(axis=0, dtype=np.float64)
            square_total += np.square(values, dtype=np.float64).sum(axis=0)
            count += values.shape[0]
    if count == 0:
        raise ValueError("cannot normalize an empty training set")
    mean = total / count
    variance = np.maximum(square_total / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def count_labels(path: str, query_ids: Iterable[int]) -> tuple[int, int]:
    positives = 0
    total = 0
    with h5py.File(path, "r") as handle:
        for query_id in query_ids:
            labels = np.asarray(
                handle[query_group_name(query_id)]["labels"][:-1], dtype=np.uint8
            )
            positives += int(labels.sum())
            total += labels.size
    return positives, total - positives


class HnswSequenceDataset(Dataset):
    """One item is one complete query trace, excluding its final timestep."""

    def __init__(
        self,
        path: str,
        query_ids: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.path = path
        self.query_ids = list(query_ids)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, FEATURE_COUNT)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, FEATURE_COUNT)
        self.k, _ = validate_trace_file(path)
        self._handle: Optional[h5py.File] = None

    @property
    def input_size(self) -> int:
        return self.k * FEATURE_COUNT

    def __len__(self) -> int:
        return len(self.query_ids)

    def _open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_id = self.query_ids[index]
        group = self._open()[query_group_name(query_id)]
        features = np.asarray(group["features"][:-1], dtype=np.float32)
        labels = np.asarray(group["labels"][:-1], dtype=np.float32)
        search_progress = np.asarray(
            group["search_progress"][:-1], dtype=np.float32
        )
        node_ids = np.asarray(group["node_ids"][:-1], dtype=np.int32)
        final_ids = np.asarray(group["final_ids"], dtype=np.int32)
        if features.shape[0] == 0:
            raise ValueError(f"query {query_id}: no non-final timestep")
        features = (features - self.mean) / self.std
        features = np.ascontiguousarray(features.reshape(features.shape[0], -1))
        return (
            torch.from_numpy(features),
            torch.from_numpy(np.ascontiguousarray(labels)),
            torch.from_numpy(np.ascontiguousarray(search_progress)),
            torch.from_numpy(np.ascontiguousarray(node_ids)),
            torch.from_numpy(np.ascontiguousarray(final_ids)),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def collate_sequences(batch):
    features, labels, search_progress, node_ids, final_ids = zip(*batch)
    lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
    return (
        pad_sequence(features, batch_first=True),
        pad_sequence(labels, batch_first=True),
        pad_sequence(search_progress, batch_first=True),
        pad_sequence(node_ids, batch_first=True, padding_value=-1),
        torch.stack(final_ids),
        lengths,
    )
