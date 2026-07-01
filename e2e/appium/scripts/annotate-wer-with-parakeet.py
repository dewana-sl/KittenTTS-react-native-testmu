#!/usr/bin/env python3
"""Annotate TestMu benchmark JSON files with Parakeet ASR WER results."""

from __future__ import annotations

import argparse
import array
import base64
import io
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Any


PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
PARAKEET_FUNCTION_ID = "d3fe9151-442b-4204-a70d-5fcc597fd610"
PARAKEET_SERVER = "grpc.nvcf.nvidia.com:443"
TARGET_SAMPLE_RATE = 16000
WER_NORMALIZATION_VERSION = "kitten-domain-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--strip-audio", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def read_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def normalize_transcript_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\bkittentts\b", "kitten tts", normalized)
    return normalized


def normalize_words(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_transcript_text(text)
    )


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def compute_wer(reference_text: str, transcript: str) -> dict[str, Any]:
    reference_words = normalize_words(reference_text)
    transcript_words = normalize_words(transcript)
    distance = edit_distance(reference_words, transcript_words)
    denominator = max(1, len(reference_words))
    wer = distance / denominator
    return {
        "parakeetWer": wer,
        "parakeetWerPercent": wer * 100,
        "parakeetEditDistance": distance,
        "parakeetReferenceWordCount": len(reference_words),
        "parakeetTranscriptWordCount": len(transcript_words),
        "parakeetWerNormalization": WER_NORMALIZATION_VERSION,
    }


def decode_wav_base64(value: str) -> tuple[int, array.array]:
    wav_bytes = base64.b64decode(value)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1:
        raise ValueError(f"Expected mono WAV, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit WAV, got {sample_width * 8}-bit")

    samples = array.array("h")
    samples.frombytes(frames)
    if samples.itemsize != 2:
        raise ValueError("This Python build does not use 16-bit short samples")
    if os.sys.byteorder != "little":
        samples.byteswap()
    return sample_rate, samples


def resample_linear(
    samples: array.array, source_rate: int, target_rate: int
) -> array.array:
    if source_rate == target_rate or len(samples) <= 1:
        return samples

    target_length = max(1, round(len(samples) * target_rate / source_rate))
    resampled = array.array("h")
    source_max_index = len(samples) - 1

    for index in range(target_length):
        position = index * source_max_index / max(1, target_length - 1)
        left = int(position)
        right = min(source_max_index, left + 1)
        fraction = position - left
        value = samples[left] + (samples[right] - samples[left]) * fraction
        resampled.append(max(-32768, min(32767, round(value))))

    return resampled


def encode_wav(samples: array.array, sample_rate: int) -> bytes:
    little_endian_samples = array.array("h", samples)
    if os.sys.byteorder != "little":
        little_endian_samples.byteswap()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(little_endian_samples.tobytes())
    return buffer.getvalue()


def prepare_parakeet_wav(wav_base64: str) -> bytes:
    source_rate, samples = decode_wav_base64(wav_base64)
    resampled = resample_linear(samples, source_rate, TARGET_SAMPLE_RATE)
    return encode_wav(resampled, TARGET_SAMPLE_RATE)


def build_riva_service(api_key: str):
    import riva.client

    function_id = os.environ.get("PARAKEET_FUNCTION_ID", PARAKEET_FUNCTION_ID)
    server = os.environ.get("PARAKEET_SERVER", PARAKEET_SERVER)
    auth = riva.client.Auth(
        uri=server,
        use_ssl=True,
        metadata_args=[
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ],
    )
    service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=False,
    )
    return service, config


def transcribe(service: Any, config: Any, wav_bytes: bytes, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = service.offline_recognize(wav_bytes, config)
            results = getattr(response, "results", []) or []
            alternatives = (
                getattr(results[0], "alternatives", []) if results else []
            )
            transcript = (
                getattr(alternatives[0], "transcript", "") if alternatives else ""
            )
            transcript = str(transcript).strip()
            if not transcript:
                raise RuntimeError("Parakeet returned an empty transcript")
            return transcript
        except Exception as error:  # noqa: BLE001 - keep row-level ASR failures in the report
            last_error = error
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(last_error or "Parakeet transcription failed"))


def annotate_row_skipped(row: dict[str, Any], message: str) -> None:
    row["parakeetStatus"] = "skipped"
    row["parakeetModel"] = PARAKEET_MODEL
    row["parakeetErrorSummary"] = message


def annotate_row_failed(row: dict[str, Any], message: str) -> None:
    row["parakeetStatus"] = "failed"
    row["parakeetModel"] = PARAKEET_MODEL
    row["parakeetErrorSummary"] = message


def strip_audio(row: dict[str, Any]) -> None:
    row.pop("werAudioBase64", None)


def main() -> None:
    args = parse_args()
    files = read_json_files(args.input_dir)
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    service = None
    config = None
    setup_error = ""

    if api_key:
        try:
            service, config = build_riva_service(api_key)
        except Exception as error:  # noqa: BLE001
            setup_error = f"Could not initialize NVIDIA Riva client: {error}"
    else:
        setup_error = "NVIDIA_API_KEY is not configured."

    for file_path in files:
        report = json.loads(file_path.read_text())
        rows = report.get("rows") if isinstance(report.get("rows"), list) else []
        annotated = 0
        skipped = 0
        failed = 0

        for row in rows:
            if row.get("status") != "passed":
                continue

            if setup_error:
                annotate_row_skipped(row, setup_error)
                skipped += 1
            elif not row.get("werAudioBase64"):
                annotate_row_skipped(row, "Benchmark row did not include WAV audio.")
                skipped += 1
            else:
                try:
                    wav_bytes = prepare_parakeet_wav(row["werAudioBase64"])
                    transcript = transcribe(service, config, wav_bytes, args.retries)
                    row["parakeetTranscript"] = transcript
                    row["parakeetStatus"] = "passed"
                    row["parakeetModel"] = PARAKEET_MODEL
                    row["parakeetSampleRate"] = TARGET_SAMPLE_RATE
                    reference_text = (
                        row.get("werReferenceText") or report.get("sampleText") or ""
                    )
                    row.update(compute_wer(reference_text, transcript))
                    row.pop("parakeetErrorSummary", None)
                    annotated += 1
                except Exception as error:  # noqa: BLE001
                    annotate_row_failed(row, str(error))
                    failed += 1

            if args.strip_audio:
                strip_audio(row)

        report["parakeetWer"] = {
            "model": PARAKEET_MODEL,
            "backend": "nvidia-riva-nim",
            "status": (
                "passed"
                if annotated and not failed
                else "skipped"
                if skipped and not failed
                else "failed"
                if failed
                else "unavailable"
            ),
            "annotatedRows": annotated,
            "skippedRows": skipped,
            "failedRows": failed,
            "errorSummary": setup_error or None,
        }
        file_path.write_text(f"{json.dumps(report, indent=2)}\n")


if __name__ == "__main__":
    main()
