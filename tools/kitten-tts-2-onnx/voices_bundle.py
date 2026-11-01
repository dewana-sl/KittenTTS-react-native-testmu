"""The built-in voices, precomputed for the on-device runtime, in one file.

For each voice the Python package derives three things at run time: the
language model's prompt prefix (the reference transcript and its codec
tokens), the 512-dim speaker embedding, and the decoder's conditioning from
the reference clip (its codec tokens, mel frames and x-vector). All three are
fixed for a built-in voice, so they are computed here once, with the Python
package's own code, and the device only reads them.

    bytes 0-7    b"KTVOICE1"
    bytes 8-11   header length, uint32 little-endian
    header       JSON: {"version", "decoder", "voices": [...]}
    data         little-endian arrays, each starting on a 4-byte boundary

Each voice in the header names its arrays as [byte offset into data, length]:
`prefix` int32 token ids, `speaker` float32[512], `prompt_token` int32,
`prompt_feat` float32[frames * 80] and `embedding` float32[192].
"""

import json
import re
import struct

import numpy as np

MAGIC = b"KTVOICE1"


def voice_language(voice_id):
    """'156_news_es' -> 'es', 'hi_hindi' -> 'hi'; the English voices carry no code."""
    match = re.search(r"_([a-z]{2})$", voice_id) or re.match(r"([a-z]{2})_[a-z]+$", voice_id)
    return match.group(1) if match else "en"


def write_voices(path, repo_dir, config, tokenizer, codec, decoder="default"):
    from kittenml.kittentts2.prompt import build_reference_prefix
    from kittenml.kittentts2.tokens import TokenMap

    token_map = TokenMap.from_config(config)
    voices_path = repo_dir / config.get("voices", "voices/voices.json")
    entries = json.loads(voices_path.read_text())
    data = bytearray()
    header = {"version": 1, "decoder": decoder, "voices": []}

    def put(array, dtype):
        while len(data) % 4:
            data.append(0)
        array = np.ascontiguousarray(np.asarray(array).reshape(-1), dtype=dtype)
        offset = len(data)
        data.extend(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
        return [offset, int(array.size)]

    for name, spec in entries.items():
        artifacts = np.load(voices_path.parent / spec["artifacts"])
        transcript = spec["transcript"].strip()
        prefix = build_reference_prefix(token_map, tokenizer.encode(transcript, add_special_tokens=False),
                                        artifacts["reference_tokens"].tolist())
        cond = codec.reference_conditioning(str(voices_path.parent / spec["reference"]))
        feat = cond["prompt_feat"][0].float().numpy()
        header["voices"].append({
            "name": name,
            "id": spec["voice_id"],
            "language": voice_language(spec["voice_id"]),
            "transcript": transcript,
            "prefix": put(prefix, np.int32),
            "speaker": put(artifacts["embedding"], np.float32),
            "prompt_token": put(cond["prompt_token"][0].cpu().numpy(), np.int32),
            "prompt_feat": put(feat, np.float32),
            "embedding": put(cond["embedding"][0].float().cpu().numpy(), np.float32),
        })

    blob = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-(len(MAGIC) + 4 + len(blob)) % 4)
    with open(path, "wb") as handle:
        handle.write(MAGIC + struct.pack("<I", len(blob)) + blob + bytes(data))
    return [voice["name"] for voice in header["voices"]]
