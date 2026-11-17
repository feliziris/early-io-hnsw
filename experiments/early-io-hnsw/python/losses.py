"""Losses for node-wise prefetch decisions over complete query sequences."""

import math

import torch
import torch.nn.functional as F


def weighted_bce_with_tp_delay(
    logits, labels, search_progress, node_ids, final_ids, lengths,
    threshold=0.5, delay_weight=1.0,
):
    """Return (total, BCE, delay), averaging each normalized query equally.

    BCE averages valid slots within each timestep, then valid timesteps.
    Delay tracks probabilities by node ID (not rank), carrying the running
    maximum through any absence. Only times from first entry are penalized.
    search_progress is the stored pop_count/efSearch, without renormalization.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("delay loss requires 0 < threshold <= 1")
    if not math.isfinite(delay_weight) or delay_weight < 0:
        raise ValueError("delay_weight must be finite and non-negative")
    if logits.ndim != 3 or logits.shape != labels.shape or logits.shape != node_ids.shape:
        raise ValueError("logits, labels and node_ids must have matching [batch,time,k] shapes")
    batch, time_count, k = logits.shape
    if (not batch or not k or final_ids.shape != (batch, k)
            or search_progress.shape != (batch, time_count) or lengths.shape != (batch,)):
        raise ValueError("invalid sequence or final_ids shape")
    if bool(((lengths <= 0) | (lengths > time_count)).any()):
        raise ValueError("each length must be in [1, time_count]")

    time_valid = torch.arange(time_count, device=logits.device)[None, :] < lengths[:, None]
    slot_valid = time_valid.unsqueeze(-1) & (node_ids >= 0)
    weights = labels + (1 - labels) * (1 + search_progress.unsqueeze(-1))
    bce_slots = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights
    timestep_bce = bce_slots.masked_fill(~slot_valid, 0).sum(-1) / slot_valid.sum(-1).clamp_min(1)
    bce = (timestep_bce.sum(-1) / lengths).mean()

    probabilities = logits.sigmoid()
    delays = []
    # Build ID matches per query to avoid a batch*time*k*k floating tensor.
    for b in range(batch):
        count = int(lengths[b])
        ids = node_ids[b, :count]
        matches = ((ids.unsqueeze(-1) == final_ids[b][None, None, :])
                   & (ids.unsqueeze(-1) >= 0))
        per_node = torch.where(matches, probabilities[b, :count, :, None], 0).amax(dim=1)
        running_max = per_node.cummax(dim=0).values
        has_entered = matches.any(dim=1).cumsum(dim=0) > 0
        penalties = F.relu(threshold - running_max) / threshold
        delays.append((penalties * has_entered).sum() / (k * count))
    delay = torch.stack(delays).mean()
    return bce + delay_weight * delay, bce, delay
