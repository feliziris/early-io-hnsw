#!/usr/bin/env python3
"""Train and test the HNSW LSTM using the embedded deterministic 80:20 split."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import (
    HnswSequenceDataset,
    collate_sequences,
    count_labels,
    fit_normalization,
    read_split,
)
from model import HnswLstm
from schema import FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-ratio", type=float, default=0.1,
                        help="fraction of the embedded 80%% train split reserved for validation")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers; keep 0 in restricted containers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefetch-threshold", type=float, default=0.5,
                        help="first score at or above this value triggers one prefetch per node")
    parser.add_argument("--min-prefetch-recall", type=float, default=0.95,
                        help="minimum validation unique-prefetch recall for best-model selection")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_train_validation(query_ids, validation_ratio: float, seed: int):
    """Deterministically split only the embedded train IDs; test IDs stay untouched."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    values = np.asarray(query_ids, dtype=np.int64).copy()
    if values.size < 2:
        raise ValueError("at least two embedded train queries are required")
    np.random.default_rng(seed).shuffle(values)
    validation_count = max(1, int(round(values.size * validation_ratio)))
    if validation_count >= values.size:
        raise ValueError("validation split leaves no training queries")
    return values[validation_count:].tolist(), values[:validation_count].tolist()


def valid_mask(lengths: torch.Tensor, time_count: int) -> torch.Tensor:
    return torch.arange(time_count, device=lengths.device)[None, :] < lengths[:, None]


def io_aware_loss(logits, labels, search_progress, lengths):
    """Masked I/O-aware BCE: y=1 weight 1, y=0 weight 1+progress."""
    valid = valid_mask(lengths, logits.shape[1]).unsqueeze(-1).float()
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    progress = search_progress.unsqueeze(-1)
    weights = labels + (1.0 - labels) * (1.0 + progress)
    # Preserve the original implementation: sum over k, then average over
    # valid query timesteps. Padded timesteps contribute exactly zero.
    return (raw * weights * valid).sum() / valid.sum()


def collect_valid(logits, labels, lengths):
    mask = valid_mask(lengths, logits.shape[1]).unsqueeze(-1).expand_as(logits)
    return logits[mask].detach().cpu(), labels[mask].detach().cpu()


def metrics_from_arrays(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    predictions = probabilities >= 0.5
    positive = labels >= 0.5
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    tn = int(np.sum(~predictions & ~positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": float(roc_auc_score(positive, probabilities)),
        "pr_auc": float(average_precision_score(positive, probabilities)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def new_prefetch_accumulator():
    return {
        "tp": 0, "fp": 0, "fn": 0,
        "delay_sum": 0, "delay_count": 0,
        "tp_delay_sum": 0, "tp_delay_count": 0,
        "fp_delay_sum": 0, "fp_delay_count": 0,
    }


def update_prefetch_metrics(
    accumulator, probabilities, node_ids, final_ids, lengths, threshold
):
    """Accumulate unique-node prefetch decisions independently per query."""
    for batch_index in range(probabilities.shape[0]):
        timestep_count = int(lengths[batch_index])
        final_set = {int(value) for value in final_ids[batch_index].tolist()}
        first_entry = {}
        prefetched = set()

        for timestep in range(timestep_count):
            for rank in range(node_ids.shape[2]):
                node_id = int(node_ids[batch_index, timestep, rank])
                if node_id < 0:
                    continue
                if node_id not in first_entry:
                    first_entry[node_id] = timestep
                if node_id in prefetched:
                    continue
                if float(probabilities[batch_index, timestep, rank]) < threshold:
                    continue

                prefetched.add(node_id)
                delay = timestep - first_entry[node_id]
                accumulator["delay_sum"] += delay
                accumulator["delay_count"] += 1
                if node_id in final_set:
                    accumulator["tp_delay_sum"] += delay
                    accumulator["tp_delay_count"] += 1
                else:
                    accumulator["fp_delay_sum"] += delay
                    accumulator["fp_delay_count"] += 1

        accumulator["tp"] += len(prefetched & final_set)
        accumulator["fp"] += len(prefetched - final_set)
        accumulator["fn"] += len(final_set - prefetched)


def finalize_prefetch_metrics(accumulator):
    tp = accumulator["tp"]
    fp = accumulator["fp"]
    fn = accumulator["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "prefetch_tp": tp,
        "prefetch_fp": fp,
        "prefetch_fn": fn,
        "prefetch_precision": precision,
        "prefetch_recall": recall,
        "prefetch_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "wasted_io_ratio": fp / max(tp + fn, 1),
        "decision_delay_mean": accumulator["delay_sum"] / max(accumulator["delay_count"], 1),
        "decision_delay_count": accumulator["delay_count"],
        "tp_decision_delay_mean": accumulator["tp_delay_sum"] / max(accumulator["tp_delay_count"], 1),
        "fp_decision_delay_mean": accumulator["fp_delay_sum"] / max(accumulator["fp_delay_count"], 1),
    }


def checkpoint_selection_key(metrics, min_prefetch_recall):
    """Higher tuple is better; enforce recall before optimizing I/O and delay."""
    recall = metrics["prefetch_recall"]
    wasted = metrics["wasted_io_ratio"]
    delay = metrics["tp_decision_delay_mean"]
    eligible = recall >= min_prefetch_recall
    if eligible:
        # Eligible epochs: minimize wasted I/O, then TP decision delay.
        return (1, -wasted, -delay, recall)
    # Fallback if no epoch reaches the recall constraint: maximize recall first.
    return (0, recall, -wasted, -delay)


def run_epoch(
    model, loader, device, optimizer=None, collect_metrics=False,
    prefetch_threshold=0.5,
):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    query_count = 0
    all_logits, all_labels = [], []
    prefetch_accumulator = new_prefetch_accumulator() if collect_metrics else None
    for features, labels, search_progress, node_ids, final_ids, lengths in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        search_progress = search_progress.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, lengths)
            loss = io_aware_loss(logits, labels, search_progress, lengths)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = features.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        query_count += batch_size
        if collect_metrics:
            valid_logits, valid_labels = collect_valid(logits, labels, lengths)
            all_logits.append(valid_logits)
            all_labels.append(valid_labels)
            update_prefetch_metrics(
                prefetch_accumulator,
                logits.detach().sigmoid().cpu(),
                node_ids,
                final_ids,
                lengths.cpu(),
                prefetch_threshold,
            )
    result = {"loss": loss_sum / max(query_count, 1)}
    if collect_metrics:
        result.update(metrics_from_arrays(
            torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()
        ))
        result.update(finalize_prefetch_metrics(prefetch_accumulator))
    return result


def save_checkpoint(path, model, mean, std, epoch, metadata):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format_version": 1, "model": "HnswLstm",
        "model_config": model.config(), "state_dict": model.state_dict(),
        "feature_names": list(FEATURE_NAMES), "feature_mean": mean,
        "feature_std": std, "epoch": epoch,
        "loss": "io_aware", "metadata": metadata,
    }, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.prefetch_threshold <= 1.0:
        raise ValueError("prefetch-threshold must be between 0 and 1")
    if not 0.0 <= args.min_prefetch_recall <= 1.0:
        raise ValueError("min-prefetch-recall must be between 0 and 1")
    seed_everything(args.seed)
    device = select_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    embedded_train_ids, test_ids = read_split(args.data)
    train_ids, validation_ids = split_train_validation(
        embedded_train_ids, args.validation_ratio, args.seed
    )
    mean, std = fit_normalization(args.data, train_ids)
    positives, negatives = count_labels(args.data, train_ids)
    train_data = HnswSequenceDataset(args.data, train_ids, mean, std)
    validation_data = HnswSequenceDataset(args.data, validation_ids, mean, std)
    test_data = HnswSequenceDataset(args.data, test_ids, mean, std)
    loader_options = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_sequences, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = HnswLstm(
        train_data.input_size, train_data.k, args.hidden_size,
        args.num_layers, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    metadata = {
        "data": str(Path(args.data).resolve()), "seed": args.seed,
        "loss": "io_aware", "prefetch_threshold": args.prefetch_threshold,
        "min_prefetch_recall": args.min_prefetch_recall,
        "selection_rule": (
            "recall_constraint_then_min_wasted_io_then_min_tp_decision_delay"
        ),
        "train_queries": len(train_ids),
        "validation_queries": len(validation_ids),
        "test_queries": len(test_ids),
        "train_positive_labels": positives, "train_negative_labels": negatives,
    }
    print(
        f"device={device} k={train_data.k} input_size={train_data.input_size} "
        f"train={len(train_ids)} validation={len(validation_ids)} "
        f"test={len(test_ids)} loss=io_aware "
        f"min_prefetch_recall={args.min_prefetch_recall:.3f}",
        flush=True,
    )

    history = []
    best_selection_key = None
    best_validation = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            collect_metrics=False,
        )
        validation_result = run_epoch(
            model, validation_loader, device, collect_metrics=True,
            prefetch_threshold=args.prefetch_threshold,
        )
        scheduler.step()
        record = {
            "epoch": epoch, "train_loss": train_result["loss"],
            "validation": validation_result,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_result['loss']:.6f} "
            f"validation_loss={validation_result['loss']:.6f} "
            f"validation_roc_auc={validation_result['roc_auc']:.6f} "
            f"prefetch_recall={validation_result['prefetch_recall']:.6f} "
            f"wasted_io={validation_result['wasted_io_ratio']:.6f} "
            f"tp_delay={validation_result['tp_decision_delay_mean']:.6f} "
            f"lr={record['learning_rate']:.8f}", flush=True,
        )
        save_checkpoint(
            output / "latest.pt", model, mean, std, epoch,
            {**metadata, "validation": validation_result},
        )
        selection_key = checkpoint_selection_key(
            validation_result, args.min_prefetch_recall
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_validation = validation_result
            best_epoch = epoch
            save_checkpoint(
                output / "best.pt", model, mean, std, epoch,
                {**metadata, "validation": validation_result},
            )

    best_checkpoint = torch.load(
        output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    test_result = run_epoch(
        model, test_loader, device, collect_metrics=True,
        prefetch_threshold=args.prefetch_threshold,
    )
    print(
        f"best_epoch={best_epoch} "
        f"best_prefetch_recall={best_validation['prefetch_recall']:.6f} "
        f"best_wasted_io={best_validation['wasted_io_ratio']:.6f} "
        f"best_tp_delay={best_validation['tp_decision_delay_mean']:.6f}",
        flush=True,
    )
    print("test=" + json.dumps(test_result, sort_keys=True), flush=True)
    save_checkpoint(
        output / "best.pt", model, mean, std, best_epoch,
        {**metadata, "validation": best_checkpoint["metadata"]["validation"],
         "test": test_result},
    )
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "arguments": vars(args), "metadata": metadata,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "loss": "io_aware", "history": history,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "selection_rule": metadata["selection_rule"],
            "test": test_result,
        }, handle, indent=2)
    with (output / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "train_query_ids": train_ids,
                   "validation_query_ids": validation_ids,
                   "test_query_ids": test_ids}, handle, indent=2)


if __name__ == "__main__":
    main()
