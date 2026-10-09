# tests/gpu/test_stream_graph.py
"""Validate streaming Flow CUDA Graph dispatch, continuity, and similarity.

A fixed 25-token planner crosses graph buckets and eventually reaches eager
fallback.  Dense graph attention is not bit-exact with eager ragged attention,
so the numerical gate is per-chunk Mel correlation above 0.95.  Packed B=1
must use the same graph path.  Run with
``pytest tests/gpu/test_stream_graph.py -m gpu -v -s``.
"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

TOKENS = [(i * 29) % 6561 for i in range(260)]
# A three-second reference spans both buckets and then reaches eager fallback.
BUCKETS = [384, 512]
_MEL_CORR_MIN = 0.95


@pytest.fixture(scope="module")
def env():
    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx")
    t2w = CosyVoice3Token2Wav(model_dir, estimator_mode="flashinfer",
                              stream_graph_buckets=BUCKETS)
    torch.manual_seed(0)
    ref_wav = torch.randn(3 * 16000) * 0.05
    cond = frontend.process(ref_wav, 16000)
    return t2w, cond


def _new_session(cond):
    return StreamSession(
        cond=cond,
        planner=ChunkPlanner(len(cond.prompt_tokens_flow),
                             chunk_size=25, scale=1),  # uniform-25
        tokens=list(TOKENS))


def _next_plan(session):
    plan = session.planner.next_chunk(len(session.tokens), finished=False)
    if plan is None:
        plan = session.planner.next_chunk(len(session.tokens), finished=True)
    return plan


def _run(t2w, cond, batched=False):
    """Return the session, waveform chunks, and newly appended Mel per step."""
    session, out, mels, prev = _new_session(cond), [], [], 0
    while True:
        plan = _next_plan(session)
        if plan is None:
            return session, out, mels
        if batched:
            out.append(t2w.stream_step_batched([session], [plan])[0])
        else:
            out.append(t2w.stream_step(session, plan))
        mels.append(session.mel_cache[:, :, prev:].clone())
        prev = session.mel_cache.shape[2]


def _corr(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (torch.dot(a - a.mean(), b - b.mean())
            / (a.std() * b.std() * (a.numel() - 1))).item()


@pytest.mark.gpu
def test_stream_graph_runs_and_matches_eager(env):
    t2w, cond = env
    est = t2w.flow.decoder.estimator
    assert est.stream_graph_buckets == BUCKETS

    # Disable only graph dispatch to obtain an eager baseline on the same model.
    est.stream_graph_buckets = None
    try:
        _, ref, ref_mels = _run(t2w, cond)
    finally:
        est.stream_graph_buckets = BUCKETS

    session, got, got_mels = _run(t2w, cond)

    # Verify continuity and that both configured graph buckets were captured.
    assert len(got) == len(ref) and len(got) >= 8
    total = sum(c.shape[1] for c in got)
    assert total == session.speech_offset, "样本数应与 speech_offset 一致"
    skeys = [k for k in est._graph_cache if k[0] == "sbucket"]
    assert sorted(k[2] for k in skeys) == BUCKETS, (
        f"graph 分支未按预期命中两桶: {skeys}")
    for c in got:
        assert torch.isfinite(c).all(), "chunk 出现 NaN/Inf"

    # Compare per-chunk Mel correlation and report observed differences.
    worst_corr, worst_mel, worst_wav = 1.0, 0.0, 0.0
    for j, (gm, rm) in enumerate(zip(got_mels, ref_mels, strict=True)):
        assert gm.shape == rm.shape, f"chunk {j}: {gm.shape} vs {rm.shape}"
        c = _corr(gm, rm)
        worst_corr = min(worst_corr, c)
        worst_mel = max(worst_mel, (gm - rm).abs().max().item())
        assert c > _MEL_CORR_MIN, (
            f"chunk {j}: mel corr {c:.4f} <= {_MEL_CORR_MIN}")
    for g, r in zip(got, ref, strict=True):
        assert g.shape == r.shape
        worst_wav = max(worst_wav, (g - r).abs().max().item())
    print(f"[stream_graph vs eager] chunks={len(got)} "
          f"worst_mel_corr={worst_corr:.4f} worst_mel_maxdiff={worst_mel:.3e} "
          f"worst_wav_maxdiff={worst_wav:.3e} (wav 逐点不比对，见 docstring)")

    # Packed B=1 uses the same graph dispatch and correlation threshold.
    _, _, mels_b = _run(t2w, cond, batched=True)
    assert len(mels_b) == len(got_mels)
    worst_b = min(_corr(mb, mg)
                  for mb, mg in zip(mels_b, got_mels, strict=True))
    print(f"[batched B=1 vs serial, graphs-on] worst_mel_corr={worst_b:.4f}")
    assert worst_b > _MEL_CORR_MIN
