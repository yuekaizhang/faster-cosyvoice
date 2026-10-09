# tests/gpu/test_interleave.py
"""Verify bit-exact session interleaving on one Torch Token2Wav instance.

Mutable state must remain inside StreamSession, so interleaved output for each
session equals its isolated run.  Run with
``pytest tests/gpu/test_interleave.py -m gpu -v``.
"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

# Deterministic synthetic speech-token streams in the 6561-entry FSQ range.
TOKENS_A = [(i * 37) % 6561 for i in range(120)]
TOKENS_B = [(i * 53) % 6561 for i in range(140)]


def _chunks(t2w, cond, tokens):
    """Drive the planner through all tokens and yield each audio chunk."""
    session = StreamSession(
        cond=cond, planner=ChunkPlanner(len(cond.prompt_tokens_flow)),
        tokens=list(tokens))
    while True:
        plan = session.planner.next_chunk(len(tokens), finished=False)
        if plan is None:
            break
        yield t2w.stream_step(session, plan)
    plan = session.planner.next_chunk(len(tokens), finished=True)
    if plan is not None:
        yield t2w.stream_step(session, plan)


@pytest.mark.gpu
def test_interleaved_sessions_bit_exact_vs_solo():
    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx")
    t2w = CosyVoice3Token2Wav(model_dir, estimator_mode="torch")

    torch.manual_seed(0)
    ref_wav = torch.randn(3 * 16000) * 0.05  # Three-second deterministic reference.
    cond = frontend.process(ref_wav, 16000)

    # Isolated baselines with fresh sessions.
    solo_a = torch.cat(list(_chunks(t2w, cond, TOKENS_A)), dim=1)
    solo_b = torch.cat(list(_chunks(t2w, cond, TOKENS_B)), dim=1)

    # Interleave both sessions chunk by chunk on one model instance.
    gens = {"a": _chunks(t2w, cond, TOKENS_A),
            "b": _chunks(t2w, cond, TOKENS_B)}
    parts = {"a": [], "b": []}
    while gens:
        for key in list(gens):
            try:
                parts[key].append(next(gens[key]))
            except StopIteration:
                del gens[key]

    assert len(parts["a"]) > 1 and len(parts["b"]) > 1, "应产生多个 chunk"
    inter_a = torch.cat(parts["a"], dim=1)
    inter_b = torch.cat(parts["b"], dim=1)
    assert torch.equal(inter_a, solo_a), "session A 交错输出 != 独跑"
    assert torch.equal(inter_b, solo_b), "session B 交错输出 != 独跑"
