# tests/test_session.py
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.streaming.chunker import ChunkPlanner


def test_session_defaults():
    s = StreamSession(cond=object(), planner=ChunkPlanner(0))
    assert s.tokens == [] and s.mel_cache is None
    assert s.speech_offset == 0 and s.chunk_index == 0
