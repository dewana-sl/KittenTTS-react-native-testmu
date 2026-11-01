"""Kitten TTS 2's language model as an ONNX graph for ONNX Runtime.

The checkpoint's linear weights are ternary: inside every 128-wide group each
value is one of {-scale, 0, +scale}. ONNX Runtime's MatMulNBits multiplies by
4-bit codes as (code - 8) * scale, so the codes 7, 8 and 9 hold -scale, 0 and
+scale exactly. The weights the graph multiplies by are bit-for-bit the
published ones, read as 4 bits each, the same trick the MLX backend uses.

The graph runs one step of the model for a block of new tokens:

    inputs   input_ids [1, S], speaker_embedding [1, 512],
             past_key_<i> / past_value_<i> [1, 1 + P, 8, 128]
    outputs  logits [1, H] for the last position, over ids head_start..vocab
             new_key_<i> / new_value_<i> [1, S, 8, 128]

Only the new keys and values come back. The caller keeps the cache and passes
its first rows in, so a step costs a copy of one token's keys, not of the
whole cache. The layout is positions first, which keeps any prefix of a
preallocated cache a contiguous tensor. Row 0 of the cache is never attended
to: it is there so that no input is ever empty, which some runtimes reject,
and the P real positions follow it.

At position 0 the token embedding is replaced by the projected speaker
embedding, as the Torch backend's forward hook does at prefill.
"""

import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TRITS_PER_BYTE = 5
_POWERS = np.array([3 ** i for i in range(TRITS_PER_BYTE)], dtype=np.uint8)


def bf16_to_f32(raw):
    """bfloat16 bits (uint16) -> float32, exactly."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_packed_checkpoint(path):
    """The packed safetensors file as numpy arrays, bf16 kept as raw bits."""
    header_len = int.from_bytes(open(path, "rb").read(8), "little")
    with open(path, "rb") as handle:
        handle.seek(8)
        header = json.loads(handle.read(header_len))
    metadata = header.pop("__metadata__", {})
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, spec in header.items():
        start, end = spec["data_offsets"]
        dtype = {"BF16": np.uint16, "U8": np.uint8, "F32": np.float32}[spec["dtype"]]
        tensors[name] = (spec["dtype"], np.frombuffer(data[start:end], dtype=dtype).reshape(spec["shape"]))
    return tensors, metadata


def ternary_to_nbits(codes, scale_bits, rows, columns, group):
    """Base-3 bytes -> MatMulNBits' [N, blocks, 64] nibbles and [N * blocks] scales."""
    trits = (codes.reshape(-1, 1) // _POWERS) % 3
    trits = trits.reshape(-1)[: rows * columns].reshape(rows, columns // group, group)
    nibbles = (trits + 7).astype(np.uint8)            # zero point 8: 7, 8, 9 = -s, 0, +s
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    scales = bf16_to_f32(scale_bits).reshape(-1)
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


class GraphBuilder:
    def __init__(self):
        self.nodes, self.initializers, self.counter = [], [], 0

    def name(self, prefix):
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def const(self, value, name=None, dtype=None):
        array = np.asarray(value, dtype=dtype)
        name = name or self.name("const")
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def op(self, op_type, inputs, outputs=None, domain="", **attrs):
        outputs = outputs or [self.name(op_type.lower())]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, domain=domain or None, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs


def build_lm(checkpoint, lm_config, kitten_config, out_dir, accuracy_level=0):
    tensors, metadata = read_packed_checkpoint(checkpoint)
    assert metadata.get("format") == "kittenml-ternary-1.58", metadata.get("format")
    group = int(metadata["group_size"])
    packed = json.loads(metadata["packed"])

    hidden = lm_config["hidden_size"]
    n_layers = lm_config["num_hidden_layers"]
    n_heads = lm_config["num_attention_heads"]
    n_kv = lm_config["num_key_value_heads"]
    head_dim = lm_config["head_dim"]
    eps = lm_config["rms_norm_eps"]
    theta = lm_config["rope_theta"]
    vocab = lm_config["vocab_size"]
    head_start = kitten_config["token_map"]["speech_start_id"]

    g = GraphBuilder()

    def dense(name):
        dtype, array = tensors[name]
        return bf16_to_f32(array) if dtype == "BF16" else array

    def linear(x, name, rows, columns):
        nbits, scales = ternary_to_nbits(tensors[f"{name}::codes"][1], tensors[f"{name}::scale"][1],
                                         rows, columns, group)
        b = g.const(nbits, f"{name}.q4")
        s = g.const(scales, f"{name}.scales")
        return g.op("MatMulNBits", [x, b, s], domain="com.microsoft", K=columns, N=rows, bits=4,
                    block_size=group, accuracy_level=accuracy_level)

    def rms_norm(x, weight_name):
        w = g.const(dense(weight_name), weight_name)
        return g.op("SimplifiedLayerNormalization", [x, w], axis=-1, epsilon=eps, stash_type=1)

    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "S"]),
              helper.make_tensor_value_info("speaker_embedding", TensorProto.FLOAT, [1, 512])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, vocab - head_start])]
    for i in range(n_layers):
        for kind in ("key", "value"):
            inputs.append(helper.make_tensor_value_info(f"past_{kind}_{i}", TensorProto.FLOAT,
                                                        [1, "R", n_kv, head_dim]))
            outputs.append(helper.make_tensor_value_info(f"new_{kind}_{i}", TensorProto.FLOAT,
                                                         [1, "S", n_kv, head_dim]))

    # Shapes and positions.
    ids_shape = g.op("Shape", ["input_ids"])
    seq = g.op("Gather", [ids_shape, g.const(1, dtype=np.int64)], axis=0)            # S
    past_shape = g.op("Shape", ["past_key_0"])
    rows = g.op("Gather", [past_shape, g.const(1, dtype=np.int64)], axis=0)          # 1 + P
    zero = g.const(0, dtype=np.int64)
    one = g.const(1, dtype=np.int64)
    past = g.op("Sub", [rows, one])                                                   # P
    positions = g.op("Range", [past, g.op("Add", [seq, past]), one])                  # [S]

    # Token embeddings, kept in float16 on disk and widened after the lookup. Only
    # the tiniest bfloat16 values (under 8e-6) lose bits, by at most 3e-8.
    table = dense("model.embed_tokens.weight")
    embed16 = g.const(table.astype(np.float16), "embed_tokens.f16")
    embeds = g.op("Cast", [g.op("Gather", [embed16, "input_ids"], axis=0)], to=TensorProto.FLOAT)

    # Speaker head at position 0.
    w0 = g.const(dense("spk_proj.0.weight").T.copy(), "spk_proj.0.weight")
    b0 = g.const(dense("spk_proj.0.bias"), "spk_proj.0.bias")
    projected = g.op("Add", [g.op("MatMul", ["speaker_embedding", w0]), b0])
    projected = g.op("LayerNormalization", [projected, g.const(dense("spk_proj.1.weight"), "spk_proj.1.weight"),
                                             g.const(dense("spk_proj.1.bias"), "spk_proj.1.bias")],
                     axis=-1, epsilon=1e-5)
    is_first = g.op("Unsqueeze", [g.op("Equal", [positions, zero]), g.const([0, 2], dtype=np.int64)])
    x = g.op("Where", [is_first, g.op("Unsqueeze", [projected, one]), embeds])        # [1, S, hidden]

    # Rotary tables for these positions (rotate-half layout, as transformers' Qwen3).
    inv_freq = (1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))).astype(np.float32)
    freqs = g.op("Mul", [g.op("Unsqueeze", [g.op("Cast", [positions], to=TensorProto.FLOAT), one]),
                         g.const(inv_freq[None, :], "rope.inv_freq")])                # [S, D/2]
    freqs = g.op("Concat", [freqs, freqs], axis=-1)                                   # [S, D]
    cos = g.op("Unsqueeze", [g.op("Cos", [freqs]), g.const([0, 2], dtype=np.int64)])  # [1, S, 1, D]
    sin = g.op("Unsqueeze", [g.op("Sin", [freqs]), g.const([0, 2], dtype=np.int64)])
    half = head_dim // 2

    def rope(t):
        first = g.op("Slice", [t, g.const([0], dtype=np.int64), g.const([half], dtype=np.int64),
                               g.const([3], dtype=np.int64)])
        second = g.op("Slice", [t, g.const([half], dtype=np.int64), g.const([head_dim], dtype=np.int64),
                                g.const([3], dtype=np.int64)])
        rotated = g.op("Concat", [g.op("Neg", [second]), first], axis=-1)
        return g.op("Add", [g.op("Mul", [t, cos]), g.op("Mul", [rotated, sin])])

    # Mask over [S, 1 + P + S]: key row t holds position t - 1, so new token s
    # (position P + s) sees rows 1 .. 1 + P + s, and row 0 is never seen.
    keys_index = g.op("Range", [zero, g.op("Add", [rows, seq]), one])                 # [1 + P + S]
    query_row = g.op("Unsqueeze", [g.op("Add", [positions, one]), one])               # [S, 1]
    hidden_rows = g.op("Or", [g.op("Greater", [keys_index, query_row]), g.op("Equal", [keys_index, zero])])
    mask = g.op("Where", [hidden_rows, g.const(np.float32(-1e9)), g.const(np.float32(0))])

    scale = g.const(np.float32(head_dim ** -0.5))
    group_q = n_heads // n_kv
    for i in range(n_layers):
        p = f"model.layers.{i}"
        h = rms_norm(x, f"{p}.input_layernorm.weight")
        q = linear(h, f"{p}.self_attn.q_proj.weight", n_heads * head_dim, hidden)
        k = linear(h, f"{p}.self_attn.k_proj.weight", n_kv * head_dim, hidden)
        v = linear(h, f"{p}.self_attn.v_proj.weight", n_kv * head_dim, hidden)
        q = g.op("Reshape", [q, g.const([1, -1, n_heads, head_dim], dtype=np.int64)])
        k = g.op("Reshape", [k, g.const([1, -1, n_kv, head_dim], dtype=np.int64)])
        g.op("Reshape", [v, g.const([1, -1, n_kv, head_dim], dtype=np.int64)], [f"new_value_{i}"])
        q = rope(rms_norm(q, f"{p}.self_attn.q_norm.weight"))
        g.op("Identity", [rope(rms_norm(k, f"{p}.self_attn.k_norm.weight"))], [f"new_key_{i}"])

        keys = g.op("Concat", [f"past_key_{i}", f"new_key_{i}"], axis=1)               # [1, T, kv, D]
        values = g.op("Concat", [f"past_value_{i}", f"new_value_{i}"], axis=1)
        keys_t = g.op("Unsqueeze", [g.op("Transpose", [keys], perm=[0, 2, 3, 1]), g.const([2], dtype=np.int64)])
        values_t = g.op("Unsqueeze", [g.op("Transpose", [values], perm=[0, 2, 1, 3]), g.const([2], dtype=np.int64)])
        # q heads 2j and 2j + 1 share kv head j, as transformers' repeat_kv lays them out.
        qg = g.op("Reshape", [g.op("Transpose", [q], perm=[0, 2, 1, 3]),
                              g.const([1, n_kv, group_q, -1, head_dim], dtype=np.int64)])  # [1, kv, G, S, D]
        scores = g.op("Add", [g.op("Mul", [g.op("MatMul", [qg, keys_t]), scale]), mask])   # [1, kv, G, S, T]
        attended = g.op("MatMul", [g.op("Softmax", [scores], axis=-1), values_t])          # [1, kv, G, S, D]
        attended = g.op("Reshape", [attended, g.const([1, n_heads, -1, head_dim], dtype=np.int64)])
        attended = g.op("Reshape", [g.op("Transpose", [attended], perm=[0, 2, 1, 3]),
                                    g.const([1, -1, n_heads * head_dim], dtype=np.int64)])
        x = g.op("Add", [x, linear(attended, f"{p}.self_attn.o_proj.weight", hidden, n_heads * head_dim)])

        h = rms_norm(x, f"{p}.post_attention_layernorm.weight")
        gate = linear(h, f"{p}.mlp.gate_proj.weight", lm_config["intermediate_size"], hidden)
        up = linear(h, f"{p}.mlp.up_proj.weight", lm_config["intermediate_size"], hidden)
        act = g.op("Mul", [g.op("Mul", [gate, g.op("Sigmoid", [gate])]), up])
        x = g.op("Add", [x, linear(act, f"{p}.mlp.down_proj.weight", hidden, lm_config["intermediate_size"])])

    last = g.op("Gather", [x, g.op("Sub", [seq, one])], axis=1)                        # [1, hidden]
    last = rms_norm(last, "model.norm.weight")
    # The tied output head, cut to the speech and control rows the sampler draws from.
    head = g.const(np.ascontiguousarray(table[head_start:].T).astype(np.float16), "lm_head.f16")
    g.op("MatMul", [last, g.op("Cast", [head], to=TensorProto.FLOAT)], ["logits"])

    graph = helper.make_graph(g.nodes, "kitten_tts_2_lm", inputs, outputs, g.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("com.microsoft", 1)],
                              producer_name="kittentts-react-native")
    model.ir_version = 8
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out_dir / "lm.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="lm.onnx.data", size_threshold=1024)
    write_int8_graph(out_dir / "lm.onnx", out_dir / "lm-int8.onnx")
    return {"head_start": head_start, "layers": n_layers, "kv_heads": n_kv, "head_dim": head_dim}


def write_int8_graph(source, target):
    """The same graph and the same weights file, with int8 activations.

    MatMulNBits' accuracy_level 4 quantizes each row of activations to int8 per
    block before multiplying. The weights are untouched, so the two graphs share
    lm.onnx.data and the second costs only its few hundred kilobytes.
    """
    model = onnx.load(str(source), load_external_data=False)
    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            for attribute in node.attribute:
                if attribute.name == "accuracy_level":
                    attribute.i = 4
    onnx.save(model, str(target))
