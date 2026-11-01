"""Export Kitten TTS 2 for the React Native SDK's on-device runtime.

    pip install -r requirements.txt
    python export.py --out kitten-tts-2-onnx

Writes a folder the SDK downloads as-is: the language model as ONNX graphs and
as the packed checkpoint MLX reads on iOS, the two decoder graphs, the
precomputed built-in voices, the tokenizer files, the English normalization
grammars and a manifest.json that lists every file with its size and SHA-256.
Files only one backend reads are marked with it, so a device downloads one copy
of the language model. Upload the folder anywhere and point `modelBaseURL` at it, or put it
in the model repository under `onnx/`, where the SDK looks by default.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import time

FORMAT = "kittentts-react-native/kitten-tts-2-onnx"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo(repo, revision):
    path = pathlib.Path(repo)
    if path.is_dir():
        return path, None
    from huggingface_hub import HfApi, snapshot_download

    from kittenml.get_model import _kitten2_files
    from kittenml.repo import resolve_repo as resolve

    _, repo_id, config = resolve(repo, None)
    local = snapshot_download(repo_id=repo_id, revision=revision,
                              allow_patterns=_kitten2_files(config, None, "packed"))
    sha = HfApi().model_info(repo_id, revision=revision).sha
    return pathlib.Path(local), {"repo": repo_id, "revision": sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="KittenML/kitten-tts-2",
                        help="Hugging Face repository or a local copy of one")
    parser.add_argument("--revision", default=None, help="repository revision (default: main)")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--default-precision", default="int8", choices=("int8", "float32"),
                        help="the language model graph the SDK loads unless an app asks for the "
                             "other: int8 activations (about 2.5x faster on a CPU) or float32")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from decoder_mlx import export_mlx_decoder
    from decoder_onnx import export_flow, export_hift
    from lm_onnx import build_lm
    from voices_bundle import write_voices

    torch.set_grad_enabled(False)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo_dir, source = resolve_repo(args.repo, args.revision)
    config = json.loads((repo_dir / "config.json").read_text())
    lm_dir = repo_dir / config.get("lm_dir", "lm")
    lm_config = json.loads((lm_dir / "config.json").read_text())

    start = time.time()
    print("language model ...", flush=True)
    lm = build_lm(repo_dir / config["lm_packed"], lm_config, config, out)

    print("decoder ...", flush=True)
    from kittenml.kittentts2.vocoder import S3Codec
    codec = S3Codec(device="cpu")
    export_flow(codec.s3gen, out / "flow.onnx")
    export_hift(codec.s3gen, out / "hift.onnx")
    export_mlx_decoder(codec.s3gen, out / "decoder-mlx.safetensors")

    print("voices ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
    voices = write_voices(out / "voices.bin", repo_dir, config, tokenizer, codec)

    for name in ("vocab.json", "merges.txt", "added_tokens.json"):
        shutil.copyfile(lm_dir / name, out / name)
    # MLX unpacks the published ternary checkpoint itself, as the Python package's MLX backend does.
    shutil.copyfile(repo_dir / config["lm_packed"], out / "lm-ternary.safetensors")
    import kitten_text_processing
    grammars = pathlib.Path(kitten_text_processing.__file__).parent / "data" / "en"
    (out / "text" / "en").mkdir(parents=True, exist_ok=True)
    for name in ("tagger.fst.gz", "verbalizer.fst.gz", "post.fst.gz"):
        shutil.copyfile(grammars / name, out / "text" / "en" / name)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    only = {"lm.onnx": "onnx", "lm-int8.onnx": "onnx", "lm.onnx.data": "onnx", "flow.onnx": "onnx",
            "hift.onnx": "onnx", "lm-ternary.safetensors": "mlx", "decoder-mlx.safetensors": "mlx"}
    manifest = {
        "format": FORMAT,
        "version": 1,
        "source": source,
        "sample_rate": config.get("sample_rate", 24000),
        "default_voice": config.get("default_voice"),
        "default_preset": config.get("default_preset", "stable"),
        "decode_presets": config.get("decode_presets", {}),
        "generation": config.get("generation", {}),
        "token_map": config["token_map"],
        "lm": {"graphs": {"int8": "lm-int8.onnx", "float32": "lm.onnx"},
               "default_precision": args.default_precision,
               "speaker_dim": config.get("speaker_embedding_dim", 512), **lm},
        "decoder": {"name": "default", "flow": "flow.onnx", "hift": "hift.onnx", "cfm_steps": 2,
                    "silence_token": 4299, "token_mel_ratio": 2, "samples_per_frame": 480, "mels": 80},
        "mlx": {"weights": "lm-ternary.safetensors", "decoder": "decoder-mlx.safetensors",
                "config": {key: lm_config[key] for key in (
                    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "head_dim", "rope_theta", "rms_norm_eps")}},
        "tokenizer": {"vocab": "vocab.json", "merges": "merges.txt", "added_tokens": "added_tokens.json"},
        "normalizer": {"version": kitten_text_processing.__version__,
                       "en": {"tagger": "text/en/tagger.fst.gz", "verbalizer": "text/en/verbalizer.fst.gz",
                              "post": "text/en/post.fst.gz"}},
        "voices": "voices.bin",
        "files": [{"path": p.relative_to(out).as_posix(), "size": p.stat().st_size, "sha256": sha256(p),
                   **({"backend": only[p.name]} if p.name in only else {})}
                  for p in files],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for backend in ("onnx", "mlx"):
        total = sum(f["size"] for f in manifest["files"] if f.get("backend", backend) == backend)
        print(f"{backend}: {total / 1e9:.2f} GB to download")
    print(f"{len(voices)} voices, {len(files)} files in {time.time() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
