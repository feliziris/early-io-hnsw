"""Two-layer stacked LSTM for per-timestep HNSW top-k prediction."""

from __future__ import annotations

import torch
from torch import nn


class HnswLstm(nn.Module):
    def __init__(self, input_size: int, k: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_size = input_size
        self.k = k
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, k)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, k]; padded positions are masked by the loss."""
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=sequences.shape[1]
        )
        return self.classifier(output)

    def config(self) -> dict:
        return {
            "input_size": self.input_size, "k": self.k,
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
