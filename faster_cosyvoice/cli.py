"""Shared parsers for human-readable command-line values."""

import argparse
import math

MEL_FRAMES_PER_SECOND = 50


def _items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError("expected a comma-separated value list")
    return items


def _require_strictly_increasing(values: tuple[float | int, ...]) -> None:
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("bucket values must be greater than zero")
    if any(left >= right for left, right in zip(values, values[1:], strict=False)):
        raise argparse.ArgumentTypeError("bucket values must be strictly increasing")


def parse_duration_bucket_seconds(value: str) -> tuple[float, ...]:
    """Parse positive, increasing duration buckets expressed in seconds."""
    try:
        durations = tuple(float(item) for item in _items(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bucket durations must be numbers") from exc
    if any(not math.isfinite(duration) for duration in durations):
        raise argparse.ArgumentTypeError("bucket durations must be finite")
    _require_strictly_increasing(durations)
    return durations


def parse_mel_duration_bucket_seconds(value: str) -> tuple[int, ...]:
    """Parse seconds at the CLI boundary and convert them to 50 Hz Mel frames."""
    durations = parse_duration_bucket_seconds(value)
    frames = tuple(round(duration * MEL_FRAMES_PER_SECOND) for duration in durations)
    for duration, frame_count in zip(durations, frames, strict=True):
        if not math.isclose(
            duration * MEL_FRAMES_PER_SECOND,
            frame_count,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise argparse.ArgumentTypeError(
                f"{duration:g} seconds is not aligned to a 20 ms Mel frame"
            )
    _require_strictly_increasing(frames)
    return frames


def parse_mel_frame_buckets(value: str) -> tuple[int, ...]:
    """Parse the legacy Mel-frame representation used by hidden aliases."""
    try:
        frames = tuple(int(item) for item in _items(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Mel-frame buckets must be integers") from exc
    _require_strictly_increasing(frames)
    return frames
