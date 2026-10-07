"""Bounded leading-silence suppression for streamed PCM16 audio.

The onset rule deliberately matches Nari's tts-bench v1 contract: 20 ms RMS
frames, 10 ms hops, per-frame DC removal, -45 dBFS, and two consecutive active
frames.  Audio is buffered only until an onset is confirmed (or the bounded
fallback is reached), and a short pre-roll is retained to avoid clipping
unvoiced consonants.
"""
from typing import Optional

import numpy as np


def audible_start_sample(pcm: bytes, sample_rate: int = 24_000
                         ) -> Optional[int]:
    if len(pcm) % 2:
        raise ValueError("PCM16 payload must contain whole samples")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frame = round(sample_rate * 0.020)
    hop = round(sample_rate * 0.010)
    threshold = 10 ** (-45.0 / 20)
    active_start = None
    active_count = 0
    for start in range(0, max(0, len(samples) - frame + 1), hop):
        window = samples[start:start + frame]
        window = window - float(np.mean(window))
        rms = float(np.sqrt(np.mean(window * window)))
        if rms >= threshold:
            if active_count == 0:
                active_start = start
            active_count += 1
            if active_count >= 2:
                return active_start
        else:
            active_start = None
            active_count = 0
    return None


class LeadingSilenceTrimmer:
    """Gate initial chunks until audible onset, then pass through unchanged."""

    def __init__(self, sample_rate: int = 24_000, preroll_ms: float = 20.0,
                 max_trim_ms: float = 2000.0,
                 min_buffer_ms: float = 400.0):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if (preroll_ms < 0 or max_trim_ms < 0 or min_buffer_ms < 0
                or preroll_ms > max_trim_ms):
            raise ValueError("require 0 <= preroll_ms <= max_trim_ms")
        self.sample_rate = sample_rate
        self.preroll_samples = round(sample_rate * preroll_ms / 1000)
        self.max_trim_samples = round(sample_rate * max_trim_ms / 1000)
        self.min_buffer_samples = round(sample_rate * min_buffer_ms / 1000)
        self._buffer = bytearray()
        self._released = False
        self._trim_start_sample: Optional[int] = None

    def feed(self, pcm: bytes, *, final: bool = False) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM16 payload must contain whole samples")
        if self._released:
            return pcm
        self._buffer.extend(pcm)
        if self._trim_start_sample is None:
            onset = audible_start_sample(bytes(self._buffer), self.sample_rate)
            if onset is not None and onset <= self.max_trim_samples:
                self._trim_start_sample = max(0, onset - self.preroll_samples)
        if self._trim_start_sample is not None:
            playable = len(self._buffer) // 2 - self._trim_start_sample
            if playable >= self.min_buffer_samples or final:
                start = self._trim_start_sample * 2
                output = bytes(self._buffer[start:])
                self._buffer.clear()
                self._released = True
                return output
            return b""

        # Do not silently erase an unusually quiet or long intro.  Once the
        # bounded decision window is exhausted, preserve the original bytes.
        detector_frame = round(self.sample_rate * 0.020)
        buffered_samples = len(self._buffer) // 2
        if final or buffered_samples >= self.max_trim_samples + detector_frame:
            output = bytes(self._buffer)
            self._buffer.clear()
            self._released = True
            return output
        return b""
