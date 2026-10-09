# tests/gpu/test_stream_flashinfer.py
"""Validate fp16 FlashInfer Flow with a chunk-causal streaming mask.

The single-session path must produce four continuous chunks.  Interleaved
sessions must remain bit-exact with isolated runs.  Run with
``pytest tests/gpu/test_stream_flashinfer.py -m gpu -v``.
"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

# Deterministic synthetic streams in the 6561-entry FSQ range.
TOKENS_A = [(i * 37) % 6561 for i in range(120)]
TOKENS_B = [(i * 53) % 6561 for i in range(140)]


@pytest.fixture(scope="module")
def env():
    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx")
    t2w = CosyVoice3Token2Wav(model_dir, estimator_mode="flashinfer")
    torch.manual_seed(0)
    ref_wav = torch.randn(3 * 16000) * 0.05  # Three-second deterministic reference.
    cond = frontend.process(ref_wav, 16000)
    return t2w, cond


def _chunks(t2w, cond, tokens):
    """Drive the planner through all tokens and yield each audio chunk."""
    session = StreamSession(
        cond=cond, planner=ChunkPlanner(len(cond.prompt_tokens_flow)),
        tokens=list(tokens))
    while True:
        plan = session.planner.next_chunk(len(tokens), finished=False)
        if plan is None:
            break
        yield session, t2w.stream_step(session, plan)
    plan = session.planner.next_chunk(len(tokens), finished=True)
    if plan is not None:
        yield session, t2w.stream_step(session, plan)


@pytest.mark.gpu
def test_flashinfer_stream_single_session(env):
    t2w, cond = env
    parts, session = [], None
    for current_session, chunk in _chunks(t2w, cond, TOKENS_A):
        session = current_session
        assert torch.isfinite(chunk).all(), "chunk 出现 NaN/Inf"
        parts.append(chunk)
    assert len(parts) == 4, f"120 token 应产生 4 chunk，实际 {len(parts)}"
    total = sum(p.shape[1] for p in parts)
    assert total == session.speech_offset, "样本数应与 speech_offset 一致"
    assert total > 24000 * 3, f"120 token 应 >3s 音频，实际 {total} samples"


@pytest.mark.gpu
def test_flashinfer_interleaved_sessions_vs_solo(env):
    t2w, cond = env
    # Isolated baselines with fresh sessions.
    solo_a = torch.cat([c for _, c in _chunks(t2w, cond, TOKENS_A)], dim=1)
    solo_b = torch.cat([c for _, c in _chunks(t2w, cond, TOKENS_B)], dim=1)

    # Interleave both sessions chunk by chunk on one model instance.
    gens = {"a": _chunks(t2w, cond, TOKENS_A),
            "b": _chunks(t2w, cond, TOKENS_B)}
    parts = {"a": [], "b": []}
    while gens:
        for key in list(gens):
            try:
                parts[key].append(next(gens[key])[1])
            except StopIteration:
                del gens[key]

    assert len(parts["a"]) > 1 and len(parts["b"]) > 1, "应产生多个 chunk"
    inter_a = torch.cat(parts["a"], dim=1)
    inter_b = torch.cat(parts["b"], dim=1)
    # The custom-mask path has no atomics, uses fixed random noise, and stores
    # mutable state only in StreamSession.  Inspect max difference before ever
    # weakening this bit-exact invariant.
    assert torch.equal(inter_a, solo_a), (
        "session A 交错输出 != 独跑，max abs diff = "
        f"{(inter_a - solo_a).abs().max().item():.3e}")
    assert torch.equal(inter_b, solo_b), (
        "session B 交错输出 != 独跑，max abs diff = "
        f"{(inter_b - solo_b).abs().max().item():.3e}")
