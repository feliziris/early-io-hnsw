#!/usr/bin/env python3
"""Export trained HNSW LSTMs or GRUs as stateful, one-timestep ONNX models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from model import HnswGru, HnswLstm
from schema import FEATURE_COUNT, FEATURE_NAMES, query_group_name


class StreamingHnswLstm(nn.Module):
    """One online timestep with normalization and explicit recurrent state."""

    def __init__(self, model: HnswLstm, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.lstm = model.lstm
        self.classifier = model.classifier
        self.k = model.k
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, FEATURE_COUNT),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, FEATURE_COUNT),
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden_in: torch.Tensor,
        cell_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_std
        timestep = normalized.reshape(features.shape[0], 1, self.k * FEATURE_COUNT)
        output, (hidden_out, cell_out) = self.lstm(
            timestep, (hidden_in, cell_in)
        )
        probabilities = torch.sigmoid(self.classifier(output[:, 0, :]))
        return probabilities, hidden_out, cell_out


class StreamingHnswGru(nn.Module):
    """One online GRU timestep with normalization and hidden state."""

    def __init__(self, model: HnswGru, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.gru = model.gru
        self.classifier = model.classifier
        self.k = model.k
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, FEATURE_COUNT),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, FEATURE_COUNT),
        )

    def forward(
        self, features: torch.Tensor, hidden_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_std
        timestep = normalized.reshape(features.shape[0], 1, self.k * FEATURE_COUNT)
        output, hidden_out = self.gru(timestep, hidden_in)
        probabilities = torch.sigmoid(self.classifier(output[:, 0, :]))
        return probabilities, hidden_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk", type=int, nargs="+", choices=(10, 20, 50), default=(10, 20, 50)
    )
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--data-root")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--validation-timesteps", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def validation_features(path: Path, timestep_limit: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        test_ids = np.asarray(handle["test_query_ids"], dtype=np.int64)
        query_id = int(test_ids[0])
        features = np.asarray(
            handle[query_group_name(query_id)]["features"][:-1], dtype=np.float32
        )
    return np.ascontiguousarray(features[:timestep_limit])


def export_one(
    topk: int,
    checkpoint_path: Path,
    data_path: Path,
    opset: int,
    validation_timesteps: int,
    tolerance: float,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    if int(config["k"]) != topk:
        raise ValueError(f"checkpoint k={config['k']} does not match --topk={topk}")
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("checkpoint feature order does not match schema.py")

    model_name = checkpoint.get("model", "HnswLstm")
    model_classes = {"HnswLstm": HnswLstm, "HnswGru": HnswGru}
    if model_name not in model_classes:
        raise ValueError(f"unsupported checkpoint model: {model_name!r}")
    is_lstm = model_name == "HnswLstm"
    model = model_classes[model_name](**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    streaming_class = StreamingHnswLstm if is_lstm else StreamingHnswGru
    streaming = streaming_class(
        model, checkpoint["feature_mean"], checkpoint["feature_std"]
    ).eval()

    layers = int(config["num_layers"])
    hidden_size = int(config["hidden_size"])
    example_features = torch.zeros(1, topk, FEATURE_COUNT, dtype=torch.float32)
    example_hidden = torch.zeros(layers, 1, hidden_size, dtype=torch.float32)
    example_cell = torch.zeros_like(example_hidden)
    example_states = (example_hidden, example_cell) if is_lstm else (example_hidden,)
    input_names = ("features", "hidden_in", "cell_in") if is_lstm else (
        "features", "hidden_in",
    )
    output_names = ("probabilities", "hidden_out", "cell_out") if is_lstm else (
        "probabilities", "hidden_out",
    )
    output_path = checkpoint_path.parent / "model.onnx"
    torch.onnx.export(
        streaming,
        (example_features, *example_states),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    del onnx_model.metadata_props[:]
    metadata = {
        "model_type": "lstm" if is_lstm else "gru",
        "topk": str(topk),
        "feature_count": str(FEATURE_COUNT),
        "feature_names": json.dumps(FEATURE_NAMES),
        "num_layers": str(layers),
        "hidden_size": str(hidden_size),
        "checkpoint_epoch": str(checkpoint.get("epoch", -1)),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, output_path)

    raw_features = validation_features(data_path, validation_timesteps)
    timestep_count = raw_features.shape[0]
    if timestep_count == 0:
        raise ValueError("ONNX validation requires at least one non-final timestep")
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32)
    normalized = (
        (torch.from_numpy(raw_features) - mean.reshape(1, 1, -1))
        / std.reshape(1, 1, -1)
    ).reshape(1, timestep_count, -1)
    with torch.inference_mode():
        full_probabilities = torch.sigmoid(
            model(normalized, torch.tensor([timestep_count]))
        ).numpy()[0]
        states = tuple(state.clone() for state in example_states)
        step_probabilities = []
        for timestep in raw_features:
            probability, *states = streaming(
                torch.from_numpy(timestep).unsqueeze(0), *states
            )
            step_probabilities.append(probability.numpy()[0])
    step_probabilities_np = np.asarray(step_probabilities, dtype=np.float32)
    pytorch_step_error = float(
        np.max(np.abs(step_probabilities_np - full_probabilities))
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    states_np = [state.numpy().copy() for state in example_states]
    ort_probabilities = []
    for timestep in raw_features:
        feeds = dict(zip(input_names[1:], states_np))
        feeds["features"] = timestep[np.newaxis, ...]
        probability, *states_np = session.run(output_names, feeds)
        ort_probabilities.append(probability[0])
    ort_probabilities_np = np.asarray(ort_probabilities, dtype=np.float32)
    onnx_error = float(np.max(np.abs(ort_probabilities_np - step_probabilities_np)))
    if pytorch_step_error > tolerance or onnx_error > tolerance:
        raise RuntimeError(
            f"validation failed for k={topk}: PyTorch step error="
            f"{pytorch_step_error}, ONNX error={onnx_error}, tolerance={tolerance}"
        )

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "data": str(data_path.resolve()),
        "model": str(output_path.resolve()),
        "opset": opset,
        "topk": topk,
        "validation_timesteps": timestep_count,
        "pytorch_full_vs_step_max_abs_error": pytorch_step_error,
        "pytorch_step_vs_onnx_max_abs_error": onnx_error,
        **{key: int(value) if value.isdigit() else value for key, value in metadata.items()},
    }
    report_path = checkpoint_path.parent / "onnx_export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"k={topk} model={output_path} timesteps={timestep_count} "
        f"pytorch_error={pytorch_step_error:.9g} onnx_error={onnx_error:.9g}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_root = (
        Path(args.checkpoint_root).resolve()
        if args.checkpoint_root
        else project_root / "model" / "sift" / "lstm_io"
    )
    data_root = (
        Path(args.data_root).resolve()
        if args.data_root
        else project_root / "output"
    )
    for topk in args.topk:
        export_one(
            topk,
            checkpoint_root / f"k{topk}" / "best.pt",
            data_root / f"sift_query10000_k{topk}.h5",
            args.opset,
            args.validation_timesteps,
            args.tolerance,
        )


if __name__ == "__main__":
    main()
