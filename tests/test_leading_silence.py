import numpy as np

from faster_cosyvoice.server.leading_silence import LeadingSilenceTrimmer, audible_start_sample

SAMPLE_RATE = 24_000


def _tone(milliseconds: int) -> bytes:
    count = round(SAMPLE_RATE * milliseconds / 1000)
    time = np.arange(count) / SAMPLE_RATE
    return (0.1 * np.sin(2 * np.pi * 440 * time) * 32767).astype("<i2").tobytes()


def _silence(milliseconds: int) -> bytes:
    return np.zeros(round(SAMPLE_RATE * milliseconds / 1000),
                    dtype="<i2").tobytes()


def test_detector_matches_nari_sustained_window_rule():
    assert audible_start_sample(_silence(60) + _tone(100)) == 1200  # 50ms


def test_trimmer_handles_onset_split_across_stream_chunks():
    trimmer = LeadingSilenceTrimmer(preroll_ms=20, max_trim_ms=2000,
                                    min_buffer_ms=0)
    assert trimmer.feed(_silence(60)) == b""
    output = trimmer.feed(_tone(100))
    # Detector onset is 50ms, so a 20ms pre-roll removes exactly 30ms.
    assert len(output) == len(_silence(30) + _tone(100))
    assert audible_start_sample(output) == 480  # 20ms
    assert trimmer.feed(_tone(10)) == _tone(10)


def test_trimmer_preserves_inaudible_final_audio():
    original = _silence(100)
    trimmer = LeadingSilenceTrimmer()
    assert trimmer.feed(original[:1000]) == b""
    assert trimmer.feed(original[1000:], final=True) == original


def test_trimmer_falls_back_without_deleting_long_quiet_intro():
    original = _silence(50)
    trimmer = LeadingSilenceTrimmer(preroll_ms=0, max_trim_ms=20)
    assert trimmer.feed(original) == original


def test_trimmer_waits_for_minimum_first_playback_buffer():
    trimmer = LeadingSilenceTrimmer(preroll_ms=20, max_trim_ms=2000,
                                    min_buffer_ms=300)
    # 60ms silence + 100ms tone leaves only 130ms after the 30ms trim.
    assert trimmer.feed(_silence(60) + _tone(100)) == b""
    output = trimmer.feed(_tone(200))
    assert len(output) == len(_silence(30) + _tone(300))
    assert audible_start_sample(output) == 480
