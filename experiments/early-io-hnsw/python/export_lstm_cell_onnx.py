#!/usr/bin/env python3
"""Expand fixed one-step ONNX LSTMs into equivalent matrix/gate operations.

Preserves normalization, classifier, weights, and explicit h/c interfaces.
This avoids backend-specific recurrent operator synchronization. It supports
only the single-step, forward, uncoupled, default-activation model used here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def convert(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Write the converted model to a separate path")
    model = onnx.load(source)
    shape = onnx.shape_inference.infer_shapes(model)
    dimensions = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*shape.graph.input, *shape.graph.value_info, *shape.graph.output]
    }
    initializers = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    replacement = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "LSTM":
            replacement.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (attrs.get("direction", b"forward") != b"forward" or
                attrs.get("layout", 0) != 0 or attrs.get("input_forget", 0) != 0 or
                any(a in attrs for a in ("clip", "activation_alpha", "activation_beta")) or
                attrs.get("activations", [b"Sigmoid", b"Tanh", b"Tanh"]) !=
                [b"Sigmoid", b"Tanh", b"Tanh"]):
            raise ValueError(f"Unsupported LSTM attributes: {attrs}")
        if dimensions.get(node.input[0], [])[:2] != [1, 1]:
            raise ValueError("Conversion requires sequence length 1 and batch size 1")
        if (len(node.input) < 7 or node.input[4] or
                (len(node.input) > 7 and node.input[7])):
            raise ValueError("Expected explicit h/c, no sequence_lens or peepholes")
        prefix = f"pipeline_cell_{count}_"
        count += 1
        h = int(attrs["hidden_size"])

        def constant(name, array):
            full = prefix + name
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(array), full))
            return full

        axis0 = constant("axis0", np.array([0], np.int64))
        split = constant("split", np.array([h] * 4, np.int64))
        w = constant("w_transposed", initializers[node.input[1]][0].T.copy())
        r = constant("r_transposed", initializers[node.input[2]][0].T.copy())
        bias = initializers[node.input[3]][0]
        b = constant("bias", (bias[:4*h] + bias[4*h:]).copy())

        def op(kind, inputs, output, **attributes):
            name = prefix + output
            replacement.append(helper.make_node(kind, inputs, [name], name=name, **attributes))
            return name

        x = op("Squeeze", [node.input[0], axis0], "x")
        old_h = op("Squeeze", [node.input[5], axis0], "h")
        old_c = op("Squeeze", [node.input[6], axis0], "c")
        xw = op("MatMul", [x, w], "xw")
        hr = op("MatMul", [old_h, r], "hr")
        gates = op("Add", [op("Add", [xw, hr], "linear"), b], "gates")
        gate_names = [prefix + name for name in ("i_raw", "o_raw", "f_raw", "g_raw")]
        replacement.append(helper.make_node("Split", [gates, split], gate_names,
                                            name=prefix + "split_gates", axis=1))
        i = op("Sigmoid", [gate_names[0]], "i")
        o = op("Sigmoid", [gate_names[1]], "o")
        f = op("Sigmoid", [gate_names[2]], "f")
        g = op("Tanh", [gate_names[3]], "g")
        new_c = op("Add", [op("Mul", [f, old_c], "forget"),
                            op("Mul", [i, g], "input_update")], "new_c")
        new_h = op("Mul", [o, op("Tanh", [new_c], "tanh_c")], "new_h")
        replacement.extend([
            helper.make_node("Unsqueeze", [new_h, axis0], [node.output[1]], name=prefix + "h_out"),
            helper.make_node("Unsqueeze", [new_c, axis0], [node.output[2]], name=prefix + "c_out"),
            helper.make_node("Unsqueeze", [node.output[1], axis0], [node.output[0]], name=prefix + "y_out"),
        ])
    if not count:
        raise ValueError("No LSTM nodes found")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    used = {name for node in model.graph.node for name in node.input}
    keep = [v for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    # Inferred annotations for removed nodes are no longer meaningful.
    del model.graph.value_info[:]
    prop = model.metadata_props.add()
    prop.key, prop.value = "recurrent_implementation", "explicit_single_step_lstm_gates"
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return count


def validate(source, destination, steps=200):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    a = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
    feature_shape = a.get_inputs()[0].shape
    state_shape = a.get_inputs()[1].shape
    rng = np.random.default_rng(42)
    h_a = np.zeros(state_shape, np.float32)
    c_a = np.zeros_like(h_a)
    h_b, c_b = h_a.copy(), c_a.copy()
    max_probability_error = max_state_error = 0.0
    for _ in range(steps):
        features = rng.uniform(0.1, 1, size=feature_shape).astype(np.float32)
        p_a, h_a, c_a = a.run(None, {"features": features, "hidden_in": h_a, "cell_in": c_a})
        p_b, h_b, c_b = b.run(None, {"features": features, "hidden_in": h_b, "cell_in": c_b})
        max_probability_error = max(max_probability_error, float(np.max(np.abs(p_a - p_b))))
        max_state_error = max(max_state_error, float(np.max(np.abs(h_a - h_b))),
                              float(np.max(np.abs(c_a - c_b))))
    if max_probability_error > 1e-4 or max_state_error > 1e-4:
        raise ValueError(f"Equivalence validation failed: {max_probability_error}, {max_state_error}")
    return {"validation_steps": steps, "seed": 42,
            "max_probability_error": max_probability_error, "max_state_error": max_state_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = convert(args.input, args.output)
    report = {"input": str(args.input), "output": str(args.output), "expanded_layers": count,
              **validate(args.input, args.output)}
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
