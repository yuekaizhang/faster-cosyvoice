import numpy as np

from examples.stream_client import _audible_offset_seconds, _playback_metrics


def test_audible_offset_matches_sustained_window_rule():
    sample_rate = 24_000
    silence = np.zeros(round(0.060 * sample_rate), dtype="<i2")
    time = np.arange(round(0.100 * sample_rate)) / sample_rate
    tone = (0.1 * np.sin(2 * np.pi * 440 * time) * 32767).astype("<i2")

    # The 20 ms analysis window first overlaps the tone at 50 ms, and two
    # consecutive active windows confirm sustained audibility.
    assert _audible_offset_seconds(silence.tobytes() + tone.tobytes()) == 0.05


def test_playback_metrics_count_gap_and_locate_audible_sample():
    audible_at, underruns = _playback_metrics(
        [(0.100, 0.050), (0.200, 0.050)], audible_offset=0.060
    )
    assert np.isclose(audible_at, 0.210)
    assert underruns == 1
