# tests/test_session.py
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession


def test_session_defaults():
    s = StreamSession(cond=object(), planner=ChunkPlanner(0))
    assert s.tokens == [] and s.mel_cache is None
    assert s.speech_offset == 0 and s.chunk_index == 0
    assert s.playback_started_at_s is None and s.emitted_duration_s == 0


def test_session_publishes_playback_credit_at_route_boundary():
    s = StreamSession(cond=object(), planner=ChunkPlanner(0))
    s.mark_pcm_routed(24000, 24000, 10.0)
    s.mark_pcm_routed(12000, 24000, 11.0)
    assert s.playback_started_at_s == 10.0
    assert s.emitted_duration_s == 1.5
