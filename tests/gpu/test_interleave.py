# tests/gpu/test_interleave.py
"""交错回归：锁死 stream_step 的承重不变量（spec D3/§6.2）——
同一个 torch-mode CosyVoice3Token2Wav 实例上多 session 交错调用，
每个 session 的输出与其独跑（solo）逐 bit 一致（torch.equal）。
状态全在 StreamSession（mel_cache/speech_offset），模块本身无跨调用状态，
重算确定性由 CausalConditionalCFM 的固定 rand_noise 保证。
运行：pytest tests/gpu/test_interleave.py -m gpu -v（容器内）。"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

# 确定性伪 token 流（值域 = FSQ 6561）
TOKENS_A = [(i * 37) % 6561 for i in range(120)]
TOKENS_B = [(i * 53) % 6561 for i in range(140)]


def _chunks(t2w, cond, tokens):
    """generator：驱动 planner 吃完 tokens，逐 chunk yield stream_step 音频。"""
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
    ref_wav = torch.randn(3 * 16000) * 0.05  # 3s 确定性噪声 ref
    cond = frontend.process(ref_wav, 16000)

    # 独跑基线（各自 fresh session，单独吃完）
    solo_a = torch.cat(list(_chunks(t2w, cond, TOKENS_A)), dim=1)
    solo_b = torch.cat(list(_chunks(t2w, cond, TOKENS_B)), dim=1)

    # 同一实例上两 session 逐 chunk 交错
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
