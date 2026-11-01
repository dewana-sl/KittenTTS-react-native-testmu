"""s3gen's flow and HiFT vocoder weights for the SDK's MLX decoder on iOS.

ios/KittenTTS2AudioDecoder.swift reads them by their torch names, so this only
puts each tensor in the form MLX computes with: weight-normed convolutions
folded to their effective weight, Conv1d weights as [out, kernel, in] and
ConvTranspose1d weights as [out, kernel, in] as well. They are stored in float16
and computed in float32: against torch, the mel and the waveform stay as close
as with float32 storage (0.026 and 0.004 at most), at half the download.
"""

from safetensors.torch import save_file
from torch import nn
from torch.nn.utils import parametrize


def mlx_decoder_state(s3gen):
    tensors = {}
    for prefix, root in (("flow", s3gen.flow), ("mel2wav", s3gen.mel2wav)):
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
    return {name: tensor.half().contiguous() for name, tensor in tensors.items()}


def export_mlx_decoder(s3gen, path):
    save_file(mlx_decoder_state(s3gen), str(path), metadata={"format": "kittentts-s3gen-mlx-1"})
