# tests/gpu/test_stream_batched.py
"""M3 Task 3 门：批量流式 flow（flow_inference_batched_streaming）+
stream_step_batched。
门 A：B=1 的 stream_step_batched vs 单请求 stream_step（同 flashinfer 实例、
      同输入、fresh session）逐 chunk 比对——理想 bit-exact（torch.equal），
      硬门 allclose(atol=1e-3)；实际相等性打印在输出里。
门 B：B=2 混合 finalize（120 vs 260 token，两 session 的 chunk 数不同，
      第 4 轮 A finalize=True 而 B finalize=False 同批）跑通，且每 session
      输出与其单独 B=1 批量运行 allclose(atol=1e-3)（bit-exact 与否打印）。
运行：pytest tests/gpu/test_stream_batched.py -m gpu -v -s（容器内）。"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

# 确定性伪 token 流（值域 = FSQ 6561；同 test_stream_flashinfer）
TOKENS_A = [(i * 37) % 6561 for i in range(120)]   # 4 chunk（最后 finalize）
TOKENS_C = [(i * 29) % 6561 for i in range(260)]   # 6 chunk → 与 A 混合 finalize


@pytest.fixture(scope="module")
def env():
    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx")
    t2w = CosyVoice3Token2Wav(model_dir, estimator_mode="flashinfer")
    torch.manual_seed(0)
    ref_wav = torch.randn(3 * 16000) * 0.05  # 3s 确定性噪声 ref
    cond = frontend.process(ref_wav, 16000)
    return t2w, cond


def _new_session(cond, tokens):
    return StreamSession(
        cond=cond, planner=ChunkPlanner(len(cond.prompt_tokens_flow)),
        tokens=list(tokens))


def _next_plan(session):
    """全部 token 已就绪的驱动：先要常规 chunk，余量不足则 finalize；
    双 None = session 完结。"""
    plan = session.planner.next_chunk(len(session.tokens), finished=False)
    if plan is None:
        plan = session.planner.next_chunk(len(session.tokens), finished=True)
    return plan


def _chunks_single(t2w, cond, tokens):
    """单请求路径：逐 chunk stream_step。"""
    session, out = _new_session(cond, tokens), []
    while True:
        plan = _next_plan(session)
        if plan is None:
            return out
        out.append(t2w.stream_step(session, plan))


def _drive_batched(t2w, cond, tokens_lists):
    """批量路径：每轮收集所有活跃 session 的 plan，一次 stream_step_batched。
    返回 (per-session chunk 列表, 是否出现过混合 finalize 批)。"""
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
                active.remove(i)  # finalize chunk 之后 session 完结
        if not idxs:
            break
        chunks = t2w.stream_step_batched(
            [sessions[i] for i in idxs], plans)
        if len(plans) > 1 and any(p.finalize for p in plans) \
                and not all(p.finalize for p in plans):
            mixed_finalize_seen = True
        for i, c in zip(idxs, chunks):
            outs[i].append(c)
    return outs, mixed_finalize_seen


def _compare_chunks(got, ref, label):
    """硬门 allclose(atol=1e-3)，同时报告是否 bit-exact 与 max diff。"""
    assert len(got) == len(ref), (
        f"{label}: chunk 数不一致 {len(got)} vs {len(ref)}")
    exact = True
    for j, (g, r) in enumerate(zip(got, ref)):
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
    """门 A：B=1 批量 == 单请求 stream_step（逐 chunk）。"""
    t2w, cond = env
    ref = _chunks_single(t2w, cond, TOKENS_A)
    (got,), _ = _drive_batched(t2w, cond, [TOKENS_A])
    assert len(ref) == 4, f"120 token 应产生 4 chunk，实际 {len(ref)}"
    _compare_chunks(got, ref, "gateA B=1 vs stream_step")


@pytest.mark.gpu
def test_gate_b_mixed_finalize_b2(env):
    """门 B：B=2 混合 finalize 跑通，且各 session == 其 B=1 单跑。"""
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
