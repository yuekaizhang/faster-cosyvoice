"""Encode incremental PCM16 audio and unknown-length streaming WAV headers."""

import struct

import numpy as np
import torch


def wav_stream_header(
    sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16
) -> bytes:
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    unknown = 0xFFFFFFFF  # streaming: RIFF/data length unknown (same as vllm-omni)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        unknown,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        unknown,
    )


def pcm16_bytes(wav: torch.Tensor) -> bytes:
    """(1, N) or (N,) fp32 [-1,1] -> little-endian int16 bytes."""
    x = wav.detach().cpu().float().numpy().reshape(-1)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
