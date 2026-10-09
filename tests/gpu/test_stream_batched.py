# tests/gpu/test_stream_batched.py
"""Validate cross-session packed streaming Flow.

A one-item packed batch must match ``stream_step`` within atol=1e-3.  A
two-item batch with mixed final/non-final chunks must match each session's
one-item packed baseline.  Run with
``pytest tests/gpu/test_stream_batched.py -m gpu -v -s``.
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
TOKENS_A = [(i * 37) % 6561 for i in range(120)]   # Four chunks; last is final.
TOKENS_C = [(i * 29) % 6561 for i in range(260)]   # Six chunks; mixed with A.


@pytest.fixture(scope="module")
def env():
    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx")
    t2w = CosyVoice3Token2Wav(model_dir, estimator_mode="flashinfer")
    torch.manual_seed(0)
    ref_wav = torch.randn(3 * 16000) * 0.05  # Three-second deterministic reference.
    cond = frontend.process(ref_wav, 16000)
    return t2w, cond


def _new_session(cond, tokens):
    return StreamSession(
        cond=cond, planner=ChunkPlanner(len(cond.prompt_tokens_flow)),
        tokens=list(tokens))


def _next_plan(session):
    """Request a normal chunk, then finalize when only a remainder is left."""
    plan = session.planner.next_chunk(len(session.tokens), finished=False)
    if plan is None:
        plan = session.planner.next_chunk(len(session.tokens), finished=True)
    return plan


def _chunks_single(t2w, cond, tokens):
    """Run one session through the serial streaming path."""
    session, out = _new_session(cond, tokens), []
    while True:
        plan = _next_plan(session)
        if plan is None:
            return out
        out.append(t2w.stream_step(session, plan))


def _drive_batched(t2w, cond, tokens_lists):
    """Run all active sessions once per packed step and collect their chunks."""
    sessions = [_new_session(cond, t) for t in tokens_lists]
    outs = [[] for _ in sessions]
    active = list(range(len(sessions)))
    mixed_finalize_seen = False
    while active:
        idxs, plans = [], []
        for i in list(active):
            plan = _next_plan(sessions[i])
            if plan is None:
                active.remove(i)
                continue
            idxs.append(i)
            plans.append(plan)
            if plan.finalize:
                active.remove(i)  # The session ends after its final chunk.
        if not idxs:
            break
        chunks = t2w.stream_step_batched(
            [sessions[i] for i in idxs], plans)
        if len(plans) > 1 and any(p.finalize for p in plans) \
                and not all(p.finalize for p in plans):
            mixed_finalize_seen = True
        for i, c in zip(idxs, chunks, strict=True):
            outs[i].append(c)
    return outs, mixed_finalize_seen


def _compare_chunks(got, ref, label):
    """Require atol=1e-3 and report whether the chunks are bit-exact."""
    assert len(got) == len(ref), (
        f"{label}: chunk 数不一致 {len(got)} vs {len(ref)}")
    exact = True
    for j, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert g.shape == r.shape, (
            f"{label} chunk {j}: shape {g.shape} vs {r.shape}")
        if not torch.equal(g, r):
            exact = False
            diff = (g - r).abs().max().item()
            assert torch.allclose(g, r, atol=1e-3), (
                f"{label} chunk {j}: max abs diff {diff:.3e} > 1e-3")
    print(f"[{label}] chunks={len(got)} bit_exact={exact}"
          + ("" if exact else " (allclose atol=1e-3 passed)"))
    return exact


@pytest.mark.gpu
def test_gate_a_b1_matches_stream_step(env):
    """A one-item packed batch matches serial ``stream_step`` per chunk."""
    t2w, cond = env
    ref = _chunks_single(t2w, cond, TOKENS_A)
    (got,), _ = _drive_batched(t2w, cond, [TOKENS_A])
    assert len(ref) == 4, f"120 token 应产生 4 chunk，实际 {len(ref)}"
    _compare_chunks(got, ref, "gateA B=1 vs stream_step")


@pytest.mark.gpu
def test_gate_b_mixed_finalize_b2(env):
    """Mixed final/non-final chunks match each session's B=1 baseline."""
    t2w, cond = env
    (solo_a,), _ = _drive_batched(t2w, cond, [TOKENS_A])
    (solo_c,), _ = _drive_batched(t2w, cond, [TOKENS_C])
    (got_a, got_c), mixed = _drive_batched(t2w, cond, [TOKENS_A, TOKENS_C])
    assert mixed, "测试设计要求出现混合 finalize 批（A 完结时 C 仍在流中）"
    assert len(solo_a) == 4 and len(solo_c) == 6, (
        f"chunk 数预期 4/6，实际 {len(solo_a)}/{len(solo_c)}")
    for chunks in (got_a, got_c):
        for c in chunks:
            assert torch.isfinite(c).all(), "chunk 出现 NaN/Inf"
    _compare_chunks(got_a, solo_a, "gateB session A (B=2 vs B=1)")
    _compare_chunks(got_c, solo_c, "gateB session C (B=2 vs B=1)")
