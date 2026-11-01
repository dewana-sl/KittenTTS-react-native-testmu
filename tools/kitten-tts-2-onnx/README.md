# Kitten TTS 2 ONNX Export

Builds the files the SDK runs Kitten TTS 2 from on the device. The source is the
same Hugging Face repository the Python package loads, and the export uses the
Python package's own code for everything it precomputes.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python export.py --out kitten-tts-2-onnx
```

It takes about a minute on a laptop and writes about 2 GB:

| File | Size | What it is |
|---|---|---|
| `lm.onnx.data` | 1.43 GB | The 1.7B speech language model's weights, with its speaker head |
| `lm-int8.onnx`, `lm.onnx` | 0.5 MB | Its graph with int8 activations, and with float32 ones |
| `lm-ternary.safetensors` | 954 MB | The packed checkpoint, copied as-is, for the SDK's MLX backend on iOS |
| `decoder-mlx.safetensors` | 271 MB | s3gen's flow and HiFT weights for the MLX decoder: weight norms folded, convolutions in MLX's layout, float16 |
| `flow.onnx` | 482 MB | s3gen's flow: codec tokens to a mel spectrogram, two meanflow steps |
| `hift.onnx` | 83 MB | s3gen's HiFT vocoder: mel spectrogram to 24 kHz audio |
| `voices.bin` | 19 MB | Every built-in voice, precomputed: prompt prefix, speaker embedding, decoder conditioning |
| `vocab.json`, `merges.txt`, `added_tokens.json` | 4.6 MB | The Qwen tokenizer, copied from the repository |
| `text/en/*.fst.gz` | 0.8 MB | kitten-text-processing's English grammars |
| `manifest.json` | | Every file with its size and SHA-256, plus the model's settings |

Files only one backend reads carry `"backend": "onnx"` or `"mlx"` in the
manifest, so a device downloads one copy of each model: 2.02 GB for ONNX
Runtime, 1.25 GB for MLX.

Upload the folder to the model repository as `onnx/`, where the SDK looks by
default, or anywhere else and pass that URL as `modelBaseURL`.

## How The Language Model Stays Exact

The checkpoint's linear weights are ternary: in every 128-wide group each value
is `-scale`, `0` or `+scale`. ONNX Runtime's `MatMulNBits` multiplies by 4-bit
codes as `(code - 8) * scale`, so the codes 7, 8 and 9 hold the three values
exactly. The graph multiplies by the published weights bit for bit, read at 4
bits each, the same trick the Python package's MLX backend uses. The token
embedding is stored in float16: of its 324M bfloat16 values, 7,928 tiny ones
(below 8e-6) move, by at most 3e-8. The output head keeps only the rows the
sampler draws from, the control tokens and the 6561 codec tokens.

`lm-int8.onnx` is the same graph with `MatMulNBits`' `accuracy_level` set to 4,
which quantizes the activations to int8 inside each matrix product; both graphs
read `lm.onnx.data`. The weights stay exact. On an M5 Pro it runs about 2.5 times
faster, and its next-token distribution is as close to float32 as the Python
package's own bfloat16 path is (mean KL 0.005 against 0.012, though single steps
reach 0.17). On 8 test sentences both graphs and the Python package said every
word right, with the same speaker similarity. `--default-precision` picks which
graph the SDK loads when an app does not ask; it is `int8` unless you change it.

## Checking An Export

```bash
python check.py --onnx kitten-tts-2-onnx
```

It runs the graphs next to the Python package in float32 (about 10 GB of
memory). For the export of revision c93cb7c:

```text
lm int8: max |logit difference| 3.94e-01, same top token at every step: True
lm float32: max |logit difference| 1.45e-04, same top token at every step: True
flow: max |mel difference| 1.24e-05
hift: max |sample difference| 8.20e-06
```

`lm_onnx.py`, `decoder_onnx.py` and `voices_bundle.py` hold the export code.
