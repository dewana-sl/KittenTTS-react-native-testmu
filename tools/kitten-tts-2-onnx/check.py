"""Compare an export with the Python package, in float32.

    python check.py --onnx kitten-tts-2-onnx

The language model graphs run a built-in voice's prompt and three more steps
next to transformers' Qwen3 with the same unpacked weights; the decoder graphs
run next to s3gen with the same noise. Needs about 10 GB of memory.
"""

import argparse
import json
import math
import pathlib
import tempfile

import numpy as np
import onnxruntime as ort
import torch


def check_language_model(folder, repo_dir, config):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from kittenml.kittentts2.prompt import build_generation_prompt, build_reference_prefix
    from kittenml.kittentts2.speaker import load_speaker_head
    from kittenml.kittentts2.tokens import TokenMap
    from kittenml.kittentts2.weights import load_lm_weights

    lm_dir = repo_dir / config.get("lm_dir", "lm")
    lm_config = json.loads((lm_dir / "config.json").read_text())
    token_map = TokenMap.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
    voices = json.loads((repo_dir / config["voices"]).read_text())
    voice = voices[config["default_voice"]]
    artifacts = np.load(repo_dir / "voices" / voice["artifacts"])
    prefix = build_reference_prefix(token_map, tokenizer.encode(voice["transcript"].strip(), add_special_tokens=False),
                                    artifacts["reference_tokens"].tolist())
    prompt = build_generation_prompt(token_map, tokenizer.encode("Hello there, this is a test.", add_special_tokens=False),
                                     reference_prefix=prefix)
    steps = [token_map.audio_id_base + int(t) for t in artifacts["reference_tokens"][:3]]

    state = {k: v.float() for k, v in load_lm_weights(repo_dir, config, "packed").items()}
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(lm_dir)), dtype=torch.float32,
                                             attn_implementation="eager")
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    model.eval()
    head = load_speaker_head(lm_config["hidden_size"], None, "cpu", torch.float32, state=state)
    with torch.no_grad():
        embeds = model.get_input_embeddings()(torch.tensor([prompt]))
        speaker = head(torch.tensor(artifacts["embedding"])[None]).unsqueeze(1)
        out = model(inputs_embeds=torch.cat([speaker, embeds[:, 1:]], dim=1), use_cache=True)
        reference = [out.logits[0, -1].numpy()]
        past = out.past_key_values
        for token in steps:
            out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            reference.append(out.logits[0, -1].numpy())
    del model, state

    manifest = json.loads((folder / "manifest.json").read_text())
    start = manifest["lm"]["head_start"]
    layers, kv, dim = (manifest["lm"][k] for k in ("layers", "kv_heads", "head_dim"))
    names = ["logits"] + [f"new_{k}_{i}" for i in range(layers) for k in ("key", "value")]
    for precision, graph in manifest["lm"]["graphs"].items():
        session = ort.InferenceSession(str(folder / graph), providers=["CPUExecutionProvider"])
        cache = {f"past_{k}_{i}": np.zeros((1, 1, kv, dim), np.float32) for i in range(layers) for k in ("key", "value")}
        worst, same_top = 0.0, True
        for ids, ref in zip([prompt] + [[t] for t in steps], reference):
            feeds = {"input_ids": np.array([ids], np.int64),
                     "speaker_embedding": artifacts["embedding"][None].astype(np.float32), **cache}
            result = dict(zip(names, session.run(names, feeds)))
            cache = {f"past_{k}_{i}": np.concatenate([cache[f"past_{k}_{i}"], result[f"new_{k}_{i}"]], axis=1)
                     for i in range(layers) for k in ("key", "value")}
            worst = max(worst, float(np.abs(result["logits"][0] - ref[start:]).max()))
            same_top &= int(result["logits"][0].argmax()) + start == int(ref.argmax())
        print(f"lm {precision}: max |logit difference| {worst:.2e}, same top token at every step: {same_top}")


def check_decoder(folder, repo_dir, config):
    from decoder_onnx import export_hift
    from kittenml.kittentts2.vocoder import S3Codec

    codec = S3Codec(device="cpu")
    voices = json.loads((repo_dir / config["voices"]).read_text())
    voice = voices[config["default_voice"]]
    ref = codec.reference_conditioning(str(repo_dir / "voices" / voice["reference"]))
    tokens = torch.tensor(np.load(repo_dir / "voices" / voice["artifacts"])["reference_tokens"][:100].astype(np.int64))
    tokens = torch.cat([tokens, torch.full((3,), 4299)])
    prompt, length = ref["prompt_token"].shape[1], tokens.shape[0]

    torch.manual_seed(7)
    with torch.inference_mode():
        mel = codec.s3gen.flow_inference(tokens, ref_dict=dict(ref), n_cfm_timesteps=2, finalize=True)
    torch.manual_seed(7)  # the same draws, in s3gen's order: the target's noise, then the whole span's
    target = torch.randn(1, 80, 2 * length)
    noise = torch.cat([torch.randn(1, 80, 2 * (prompt + length))[:, :, :2 * prompt], target], dim=2)
    flow = ort.InferenceSession(str(folder / "flow.onnx"), providers=["CPUExecutionProvider"])
    ours = flow.run(["mel"], {"speech_tokens": tokens[None].numpy(), "prompt_token": ref["prompt_token"].numpy(),
                              "prompt_feat": ref["prompt_feat"].numpy(), "embedding": ref["embedding"].numpy(),
                              "noise": noise.numpy()})[0]
    print(f"flow: max |mel difference| {np.abs(ours - mel.numpy()).max():.2e}")

    # The shipped vocoder draws its noise inside the graph, so this compares a
    # build that takes the draws as inputs.
    torch.manual_seed(11)
    with torch.inference_mode():
        wav, _ = codec.s3gen.mel2wav.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
    torch.manual_seed(11)
    phase = -math.pi + torch.rand(1, 9, 1) * 2 * math.pi
    phase[:, 0] = 0
    source_noise = torch.randn(1, 9, mel.shape[2] * 480)
    with tempfile.TemporaryDirectory() as tmp:
        export_hift(codec.s3gen, pathlib.Path(tmp) / "hift.onnx", random_inputs=True)
        hift = ort.InferenceSession(str(pathlib.Path(tmp) / "hift.onnx"), providers=["CPUExecutionProvider"])
        ours = hift.run(["wav"], {"mel": mel.numpy(), "phase": phase.numpy(), "noise": source_noise.numpy()})[0]
    print(f"hift: max |sample difference| {np.abs(ours - wav.numpy()).max():.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="the folder export.py wrote")
    parser.add_argument("--repo", default="KittenML/kitten-tts-2")
    args = parser.parse_args()

    from export import resolve_repo

    torch.set_grad_enabled(False)
    folder = pathlib.Path(args.onnx)
    repo_dir, _ = resolve_repo(args.repo, json.loads((folder / "manifest.json").read_text())
                               .get("source", {}).get("revision"))
    config = json.loads((repo_dir / "config.json").read_text())
    check_language_model(folder, repo_dir, config)
    check_decoder(folder, repo_dir, config)


if __name__ == "__main__":
    main()
