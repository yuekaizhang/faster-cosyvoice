"""Stream raw PCM, save WAV, and report Nari-compatible audible latency.

Registered voice:
  uv run python examples/stream_client.py --voice demo --target-text "Hello." --out out.wav

One-shot voice clone:
  uv run python examples/stream_client.py --ref-audio ref.wav --ref-text "Reference." \
      --target-text "Hello." --out out.wav
"""
import argparse
import base64
import json
import math
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from faster_cosyvoice.server.leading_silence import audible_start_sample

SAMPLE_RATE = 24_000
FRAME_BYTES = 2


def _audible_offset_seconds(pcm: bytes) -> float | None:
    """Apply Nari tts-bench v1's audible-onset rule."""
    start = audible_start_sample(pcm, SAMPLE_RATE)
    return None if start is None else start / SAMPLE_RATE


def _playback_metrics(observations, audible_offset):
    """Simulate immediate playback using Nari's zero-buffer policy."""
    if not observations:
        return None, 0
    deadline = observations[0][0]
    cursor = 0.0
    audible_at = None
    underruns = 0
    for index, (arrival, duration) in enumerate(observations):
        if arrival > deadline:
            if index > 0:
                underruns += 1
            deadline = arrival
        if (audible_offset is not None and audible_at is None
                and cursor <= audible_offset < cursor + duration):
            audible_at = deadline + audible_offset - cursor
        deadline += duration
        cursor += duration
    return audible_at, underruns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--voice")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--out", default="out.wav")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    request = {
        "model": "faster-cosyvoice",
        "input": args.target_text,
        "stream": True,
        "response_format": "pcm",
        "seed": args.seed,
    }
    if args.voice:
        if args.ref_audio or args.ref_text:
            parser.error("choose --voice or --ref-audio/--ref-text")
        request["voice"] = args.voice
    else:
        if not args.ref_audio or not args.ref_text:
            parser.error("provide --voice or both --ref-audio and --ref-text")
        request["ref_audio"] = (
            "data:audio/wav;base64,"
            + base64.b64encode(Path(args.ref_audio).read_bytes()).decode()
        )
        request["ref_text"] = args.ref_text

    started = time.perf_counter()
    first_body = None
    remainder = b""
    pcm_parts = []
    observations = []
    with httpx.stream(
        "POST", f"{args.url.rstrip('/')}/v1/audio/speech",
        json=request, timeout=300,
    ) as response:
        response.raise_for_status()
        for raw in response.iter_raw():
            if not raw:
                continue
            arrival = time.perf_counter() - started
            if first_body is None:
                first_body = arrival
            aligned = remainder + raw
            complete = len(aligned) - len(aligned) % FRAME_BYTES
            pcm_chunk, remainder = aligned[:complete], aligned[complete:]
            if pcm_chunk:
                pcm_parts.append(pcm_chunk)
                observations.append(
                    (arrival, len(pcm_chunk) / FRAME_BYTES / SAMPLE_RATE)
                )
    wall = time.perf_counter() - started
    if remainder:
        raise RuntimeError("response ended with a partial PCM sample")
    pcm = b"".join(pcm_parts)
    if not pcm:
        raise RuntimeError("response contained no PCM audio")
    sf.write(args.out, np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE,
             subtype="PCM_16")

    audible_offset = _audible_offset_seconds(pcm)
    audible_at, underruns = _playback_metrics(observations, audible_offset)
    result = {
        "ttfb_ms": round(
            (first_body if first_body is not None else wall) * 1000, 1
        ),
        "first_playable_ms": round(observations[0][0] * 1000, 1),
        "audible_ttfa_ms": (
            round(audible_at * 1000, 1) if audible_at is not None else None
        ),
        "leading_silence_ms": (
            round(audible_offset * 1000, 1) if audible_offset is not None else None
        ),
        "underruns": underruns,
        "audio_s": round(len(pcm) / FRAME_BYTES / SAMPLE_RATE, 2),
        "wall_s": round(wall, 2),
    }
    if any(isinstance(value, float) and not math.isfinite(value)
           for value in result.values()):
        raise RuntimeError("non-finite timing result")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
