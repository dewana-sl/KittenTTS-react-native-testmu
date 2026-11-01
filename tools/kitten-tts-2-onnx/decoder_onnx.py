"""Kitten TTS 2's audio decoder (chatterbox-turbo's s3gen) as two ONNX graphs.

    flow.onnx   codec tokens + the voice's prompt -> mel spectrogram
    hift.onnx   mel spectrogram -> 24 kHz waveform

The flow is the meanflow model in two Euler steps from t = 0 to 1, the
schedule `S3Codec.decode` runs. Its starting noise is an input, so the caller
owns the random numbers and a seeded generation can reproduce them.

The vocoder ends in an inverse STFT, which ONNX has no operator for. With a
16-point FFT it is a matrix product per frame followed by overlap-add, written
here with MatMul and ConvTranspose; the forward STFT of the harmonic source is a
strided convolution. Both are checked against torch.stft and torch.istft.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Flow(nn.Module):
    """`CausalMaskedDiffWithXvec.inference` for one utterance, finalize=True, meanflow."""

    def __init__(self, flow, steps=2):
        super().__init__()
        self.flow = flow
        self.steps = steps

    def forward(self, speech_tokens, prompt_token, prompt_feat, embedding, noise):
        flow = self.flow
        embedding = F.normalize(embedding, dim=1)
        spks = flow.spk_embed_affine_layer(embedding)
        token = torch.cat([prompt_token, speech_tokens], dim=1)
        token_len = torch.ones(1, dtype=torch.long) * token.shape[1]
        h, _ = flow.encoder(flow.input_embedding(token), token_len)
        h = flow.encoder_proj(h)                                       # [1, 2(P+T), 80]
        mel_prompt = prompt_feat.shape[1]
        conds = torch.cat([prompt_feat, torch.zeros_like(h[:, mel_prompt:])], dim=1).transpose(1, 2)
        mu = h.transpose(1, 2)
        mask = torch.ones_like(mu[:, :1])
        x = noise
        estimator = flow.decoder.estimator
        for i in range(self.steps):
            t = torch.full((1,), i / self.steps)
            r = torch.full((1,), (i + 1) / self.steps)
            x = x + (r - t) * estimator(x, mask=mask, mu=mu, t=t, spks=spks, cond=conds, r=r)
        return x[:, :, mel_prompt:]


class Hift(nn.Module):
    """`HiFTGenerator.inference` with the STFT pair in plain operators.

    With `random_inputs`, the sine phases and the source noise are inputs instead
    of being drawn inside the graph; only the parity check uses that.
    """

    def __init__(self, hift, random_inputs=False):
        super().__init__()
        self.hift = hift
        self.random_inputs = random_inputs
        n_fft, hop = hift.istft_params["n_fft"], hift.istft_params["hop_len"]
        self.n_fft, self.hop = n_fft, hop
        window = hift.stft_window.double()
        bins = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(bins, dtype=torch.float64)[:, None]
        angle = 2 * math.pi * k * n / n_fft
        # Forward: X[k] = sum x[n] w[n] e^{-i 2 pi k n / N}, real rows then imaginary rows.
        forward = torch.cat([torch.cos(angle) * window, -torch.sin(angle) * window], dim=0)
        self.register_buffer("stft_basis", forward.float()[:, None, :])          # [2B, 1, N]
        # Inverse real FFT of a one-sided spectrum, times the synthesis window.
        weight = torch.full((bins, 1), 2.0, dtype=torch.float64)
        weight[0] = weight[-1] = 1.0
        cos_part = (torch.cos(angle) * weight).T / n_fft                         # [N, B]
        sin_part = (-torch.sin(angle) * weight).T / n_fft
        sin_part[:, 0] = 0
        sin_part[:, -1] = 0
        inverse = torch.cat([cos_part, sin_part], dim=1) * window[:, None]        # [N, 2B]
        self.register_buffer("istft_basis", inverse.float())
        self.register_buffer("overlap", torch.eye(n_fft)[:, None, :])           # [N, 1, N]
        self.register_buffer("window_sq", (window ** 2).float()[None, None, :])

    def stft(self, x):
        x = F.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        return F.conv1d(x, self.stft_basis, stride=self.hop)                     # [1, 2B, frames]

    def istft(self, magnitude, phase):
        magnitude = torch.clamp(magnitude, max=1e2)
        spectrum = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        frames = torch.matmul(self.istft_basis, spectrum)                       # [1, N, frames]
        signal = F.conv_transpose1d(frames, self.overlap, stride=self.hop)
        envelope = F.conv_transpose1d(torch.ones_like(frames[:, :1]), self.window_sq, stride=self.hop)
        signal = signal / torch.clamp(envelope, min=1e-11)
        half = self.n_fft // 2
        return signal[:, 0, half:signal.shape[2] - half]

    def source(self, f0, phase=None, noise=None):
        sine_gen = self.hift.m_source.l_sin_gen
        harmonics = torch.arange(1, sine_gen.harmonic_num + 2, dtype=f0.dtype)[None, :, None]
        increments = f0 * harmonics / sine_gen.sampling_rate                    # [1, H, L]
        theta = 2 * math.pi * torch.remainder(torch.cumsum(increments, dim=-1), 1.0)
        if phase is None:
            phase = (torch.rand_like(increments[:, :, :1]) * 2 - 1) * math.pi
            phase = torch.cat([torch.zeros_like(phase[:, :1]), phase[:, 1:]], dim=1)
        sine = sine_gen.sine_amp * torch.sin(theta + phase)
        uv = (f0 > sine_gen.voiced_threshold).to(f0.dtype)
        amplitude = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
        if noise is None:
            noise = torch.randn_like(sine)
        sine = sine * uv + amplitude * noise
        merged = self.hift.m_source.l_tanh(self.hift.m_source.l_linear(sine.transpose(1, 2)))
        return merged.transpose(1, 2)                                            # [1, 1, L]

    def forward(self, mel, phase=None, noise=None):
        hift = self.hift
        f0 = hift.f0_predictor(mel)
        f0 = hift.f0_upsamp(f0[:, None])                                        # [1, 1, L]
        s = self.source(f0, phase, noise)
        s_stft = self.stft(s[:, 0])
        x = hift.conv_pre(mel)
        for i in range(hift.num_upsamples):
            x = F.leaky_relu(x, hift.lrelu_slope)
            x = hift.ups[i](x)
            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)
            si = hift.source_resblocks[i](hift.source_downs[i](s_stft))
            x = x + si
            xs = None
            for j in range(hift.num_kernels):
                y = hift.resblocks[i * hift.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / hift.num_kernels
        x = F.leaky_relu(x)
        x = hift.conv_post(x)
        bins = self.n_fft // 2 + 1
        wav = self.istft(torch.exp(x[:, :bins]), torch.sin(x[:, bins:]))
        return torch.clamp(wav, -hift.audio_limit, hift.audio_limit)


def export_flow(s3gen, path, opset=17):
    model = Flow(s3gen.flow).eval()
    tokens = torch.randint(0, 6561, (1, 40))
    prompt = torch.randint(0, 6561, (1, 30))
    feat = torch.randn(1, 60, 80)
    emb = torch.randn(1, 192)
    noise = torch.randn(1, 80, 140)
    torch.onnx.export(
        model, (tokens, prompt, feat, emb, noise), str(path), dynamo=False, opset_version=opset,
        input_names=["speech_tokens", "prompt_token", "prompt_feat", "embedding", "noise"],
        output_names=["mel"],
        dynamic_axes={"speech_tokens": {1: "T"}, "prompt_token": {1: "P"}, "prompt_feat": {1: "P2"},
                      "noise": {2: "N"}, "mel": {2: "M"}})
    return model


def export_hift(s3gen, path, opset=17, random_inputs=False):
    model = Hift(s3gen.mel2wav, random_inputs).eval()
    mel = torch.randn(1, 80, 50)
    names, args, axes = ["mel"], [mel], {"mel": {2: "M"}, "wav": {1: "L"}}
    if random_inputs:
        harmonics = s3gen.mel2wav.m_source.l_sin_gen.harmonic_num + 1
        args += [torch.zeros(1, harmonics, 1), torch.zeros(1, harmonics, 50 * 480)]
        names += ["phase", "noise"]
        axes["noise"] = {2: "L"}
    torch.onnx.export(model, tuple(args), str(path), dynamo=False, opset_version=opset,
                      input_names=names, output_names=["wav"], dynamic_axes=axes)
    return model
