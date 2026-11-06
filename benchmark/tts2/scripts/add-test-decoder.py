"""Add `stock_1step` to an exported model folder: s3gen's flow held to one step.

Test only. It is decoder-mlx.safetensors with "flow_steps": "1" in its metadata,
which the MLX decoder reads, listed in manifest.json as a decoder of its own.

    python add-test-decoder.py <folder>
"""

import hashlib
import json
import pathlib
import struct
import sys

folder = pathlib.Path(sys.argv[1])
source, target = folder / "decoder-mlx.safetensors", folder / "decoder-stock_1step-mlx.safetensors"
with open(source, "rb") as reader:
    length = struct.unpack("<Q", reader.read(8))[0]
    header = json.loads(reader.read(length))
    header.setdefault("__metadata__", {})["flow_steps"] = "1"
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(target, "wb") as writer:
        writer.write(struct.pack("<Q", len(blob)) + blob)
        while chunk := reader.read(1 << 24):
            writer.write(chunk)

digest = hashlib.sha256(target.read_bytes()).hexdigest()
manifest = json.loads((folder / "manifest.json").read_text())
manifest.setdefault("decoders", {})["stock_1step"] = {"weights": target.name, "voices": "voices.bin", "cfm_steps": 1}
manifest["files"] = [f for f in manifest["files"] if f["path"] != target.name] + [
    {"path": target.name, "size": target.stat().st_size, "sha256": digest, "backend": "native", "decoder": "stock_1step"}]
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"{target.name}: {target.stat().st_size} bytes")
