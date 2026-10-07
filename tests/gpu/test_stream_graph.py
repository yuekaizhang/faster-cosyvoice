# tests/gpu/test_stream_graph.py
"""[M3.5-r2] 流式 bucketed CUDA graphs（stream_graph_buckets）+ uniform-25 门。

(a) 跑通 + 样本数连续性：uniform-25 planner 驱动 260 伪 token，途中越过最大
    bucket → graph→eager fallback 也被覆盖；graph cache 出现 "sbucket" 条目
    = dispatch 接线证据。
(b) 数值：同实例同输入 graphs-on vs graphs-off（eager flashinfer ragged）。
    实测（H100）：单次 estimator 调用 dense-SDPA vs ragged 差 ~5-7% rel —
    与已上线的 offline bucketed graph 完全同级（0.75 max / 6.7% rel 对照
    实验）；此差经 10 步 euler ODE 放大 → mel maxdiff ~3.5、mel corr
    ~0.988，再经 hift（对 mel 局部差相位敏感）→ 波形逐点比对失去意义
    （corr ~0）。因此数值门 = 逐 chunk mel（session.mel_cache 段）相关系数
    > 0.95 + 报告实测 diff；质量仲裁与 offline 知识一致 = ASR CER 门
    （服务级，见 milestone 报告）。波形逐点 allclose 无法成立——如需逐位
    稳定请保持该旋钮关闭（默认即关）。
(c) stream_step_batched B=1（packed 批路径）同样命中 graph 分支。
运行：pytest tests/gpu/test_stream_graph.py -m gpu -v -s（容器内）。"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

TOKENS = [(i * 29) % 6561 for i in range(260)]
# 3s ref → prompt ~75 token → chunk 序列 ~200..670 帧：384/512 两桶 + 尾部
# 越界 → eager fallback 段
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
    """返回 (session, wav chunks, mel 段列表)。mel 段 = 每步 mel_cache 新增。"""
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

    # graphs-off 基线（同实例：eager 流式路径不读 stream_graph_buckets 之外
    # 的旁路状态，_forward_packed 与默认配置逐位一致）
    est.stream_graph_buckets = None
    try:
        _, ref, ref_mels = _run(t2w, cond)
    finally:
        est.stream_graph_buckets = BUCKETS

    session, got, got_mels = _run(t2w, cond)

    # (a) 跑通 + 连续性 + dispatch 接线证据
    assert len(got) == len(ref) and len(got) >= 8
    total = sum(c.shape[1] for c in got)
    assert total == session.speech_offset, "样本数应与 speech_offset 一致"
    skeys = [k for k in est._graph_cache if k[0] == "sbucket"]
    assert sorted(k[2] for k in skeys) == BUCKETS, (
        f"graph 分支未按预期命中两桶: {skeys}")
    for c in got:
        assert torch.isfinite(c).all(), "chunk 出现 NaN/Inf"

    # (b) 逐 chunk mel 相关性门 + 实测 diff 报告（见模块 docstring）
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

    # (c) packed B=1 批路径同样命中 graph 分支（与 serial 同 dispatch），
    # mel 与 serial graphs-on 相关性同门
    _, _, mels_b = _run(t2w, cond, batched=True)
    assert len(mels_b) == len(got_mels)
    worst_b = min(_corr(mb, mg)
                  for mb, mg in zip(mels_b, got_mels, strict=True))
    print(f"[batched B=1 vs serial, graphs-on] worst_mel_corr={worst_b:.4f}")
    assert worst_b > _MEL_CORR_MIN
