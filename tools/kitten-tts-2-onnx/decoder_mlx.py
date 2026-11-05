"""s3gen's flow and HiFT vocoder weights for the SDK's MLX decoder on iOS.

ios/KittenTTS2AudioDecoder.swift reads them by their torch names, so this only
puts each tensor in the form MLX computes with: weight-normed convolutions
folded to their effective weight, Conv1d weights as [out, kernel, in] and
ConvTranspose1d weights as [out, kernel, in] as well. They are stored in float16
and computed in float32: against torch, the mel and the waveform stay as close
as with float32 storage (0.026 and 0.004 at most), at half the download.

A student flow (the Python package's `decoder="student_w4"`) runs in one step and
replaces only the flow. Its quantised weights are kept as they are published, as
`<name>.codes` (int8) and `<name>.scale` (float32, per row), so codes / scale gives
the bits torch computes with, and the rest of its flow stays float32.
"""

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn.utils import parametrize


def _mlx_tensors(prefix, root):
    tensors = {}
    for name, module in root.named_modules():
        if "parametrizations" in name:
            continue
        full = f"{prefix}.{name}" if name else prefix
        for param_name, param in module.named_parameters(recurse=False):
            tensors[f"{full}.{param_name}"] = param.detach()
        if parametrize.is_parametrized(module, "weight"):
            tensors[f"{full}.weight"] = module.weight.detach()
        weight = tensors.get(f"{full}.weight")
        if isinstance(module, nn.ConvTranspose1d):
            tensors[f"{full}.weight"] = weight.permute(1, 2, 0)      # [in, out, k] -> [out, k, in]
        elif isinstance(module, nn.Conv1d):
            tensors[f"{full}.weight"] = weight.permute(0, 2, 1)      # [out, in, k] -> [out, k, in]
    return tensors


def mlx_decoder_state(s3gen):
    tensors = {**_mlx_tensors("flow", s3gen.flow), **_mlx_tensors("mel2wav", s3gen.mel2wav)}
    return {name: tensor.half().contiguous() for name, tensor in tensors.items()}


def export_mlx_decoder(s3gen, path):
    save_file(mlx_decoder_state(s3gen), str(path), metadata={"format": "kittentts-s3gen-mlx-1"})


def student_decoder_state(s3gen, student_weights):
    from kittenml.kittentts2.vocoder import _unpack4, load_student_flow

    blob = torch.load(str(student_weights), map_location="cpu", weights_only=False)
    tensors = _mlx_tensors("flow", load_student_flow(student_weights, device="cpu"))
    for name, entry in blob["quantised"].items():
        codes = _unpack4(entry["packed"], entry["numel"]).reshape(list(entry["shape"]))
        scale = entry["scale"].reshape(-1, *([1] * (codes.dim() - 1)))
        if codes.dim() == 3:
            codes = codes.permute(0, 2, 1)                           # [out, in, k] -> [out, k, in]
        full = f"flow.{name}"
        if not torch.equal(codes.float() / scale, tensors.pop(full)):
            raise ValueError(f"{full} does not match its stored codes")
        tensors[f"{full}.codes"] = codes
        tensors[f"{full}.scale"] = scale
    vocoder = {name: tensor.half() for name, tensor in _mlx_tensors("mel2wav", s3gen.mel2wav).items()}
    return {name: tensor.contiguous() for name, tensor in {**tensors, **vocoder}.items()}


def export_student_decoder(s3gen, student_weights, path):
    save_file(student_decoder_state(s3gen, student_weights), str(path),
              metadata={"format": "kittentts-s3gen-mlx-1", "flow_steps": "1"})
