import sys

import pytest
import torch

from faster_cosyvoice.token2wav.frontend import RefAudioFrontend, SpeakerCache, truncate_2to1


def test_truncate_2to1_feat_shorter():
    tokens, feat_len = truncate_2to1(list(range(100)), feat_len=60)
    assert tokens == list(range(30)) and feat_len == 60


def test_truncate_2to1_tokens_shorter():
    tokens, feat_len = truncate_2to1(list(range(10)), feat_len=100)
    assert tokens == list(range(10)) and feat_len == 20


def test_speaker_cache_key_includes_audio_and_text():
    c = SpeakerCache(max_size=2)
    wav = torch.zeros(16000)
    k1 = c.make_key(wav, "a")
    assert k1 != c.make_key(wav, "b")            # 同音频不同文本 → 不同 key
    assert k1 != c.make_key(torch.ones(16000), "a")  # 同文本不同音频 → 不同 key
    assert k1 == c.make_key(torch.zeros(16000), "a")


def test_speaker_cache_lru_eviction():
    c = SpeakerCache(max_size=2)
    c.put("k1", 1)
    c.put("k2", 2)
    assert c.get("k1") == 1        # 触碰 k1
    c.put("k3", 3)                 # 淘汰 k2
    assert c.get("k2") is None and c.get("k1") == 1 and c.get("k3") == 3


def test_campplus_trt_missing_tensorrt_fails_loud(monkeypatch):
    """campplus_trt=True 且 tensorrt 不可 import → init 即抛可操作的 RuntimeError。"""
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # import 时抛 ImportError
    with pytest.raises(RuntimeError, match="tensorrt"):
        RefAudioFrontend("nonexistent/campplus.onnx", campplus_trt=True)
