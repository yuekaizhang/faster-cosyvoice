"""Parse optional CUDA Graph bucket lists from command-line arguments."""

import argparse
import math

MEL_FRAMES_PER_SECOND = 50


def _split_csv(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",")]
    if any(not value for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated bucket values")
    return values


def _validate(values: tuple[float | int, ...]) -> None:
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("buckets must be greater than zero")
    if any(current >= following
           for current, following in zip(values, values[1:], strict=False)):
        raise argparse.ArgumentTypeError("buckets must be strictly increasing")


def parse_seconds(text: str) -> tuple[float, ...]:
    """Parse positive, increasing bucket durations such as ``8,12,16``."""
    try:
        seconds = tuple(float(value) for value in _split_csv(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("buckets must be numbers") from exc
    if any(not math.isfinite(value) for value in seconds):
        raise argparse.ArgumentTypeError("buckets must be finite")
    _validate(seconds)
    return seconds


def parse_mel_seconds(text: str) -> tuple[int, ...]:
    """Convert second-based buckets to the 50 Hz Mel-frame representation."""
    seconds = parse_seconds(text)
    frames = tuple(round(value * MEL_FRAMES_PER_SECOND) for value in seconds)
    for value, frame_count in zip(seconds, frames, strict=True):
        if not math.isclose(
            value * MEL_FRAMES_PER_SECOND,
            frame_count,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise argparse.ArgumentTypeError(
                f"{value:g} seconds is not aligned to a 20 ms Mel frame"
            )
    return frames


def parse_mel_frames(text: str) -> tuple[int, ...]:
    """Parse legacy CLI buckets that are already expressed in Mel frames."""
    try:
        frames = tuple(int(value) for value in _split_csv(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Mel-frame buckets must be integers") from exc
    _validate(frames)
    return frames
