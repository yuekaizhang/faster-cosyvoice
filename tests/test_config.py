# tests/test_config.py
from faster_cosyvoice.config import LLMConfig, ServerConfig, Token2WavConfig


def test_llm_defaults_match_spec_appendix_a():
    c = LLMConfig()
    assert c.target_model == "yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF"
    assert c.draft_model == "yuekai/cosyvoice3_llm_dspark"
    assert (c.temperature, c.top_p, c.top_k) == (0.8, 0.95, 15)
    assert c.repetition_penalty == 1.1
    assert c.max_tokens == 2048
    # 0.6：dspark draft KV/graphs 不计入 vLLM 配额 + token2wav 同卡，0.8 会 OOM
    assert c.gpu_memory_utilization == 0.6


def test_token2wav_defaults():
    c = Token2WavConfig()
    assert c.estimator_mode == "flashinfer"
    assert c.batch_size == 8
    assert c.cuda_graph_buckets is None  # 默认关：opt-in bucketed CUDA graphs
    assert c.hift_compile is False  # 默认关：opt-in torch.compile(hift.decode)
    # [M3.5-r2] 默认关：opt-in 流式 bucketed CUDA graphs
    assert c.stream_graph_buckets is None
    # [M3.5-r4] 默认关：opt-in 流式 hift bucketed CUDA graphs
    assert c.hift_graph_buckets is None


def test_server_chunk_defaults_match_current_behavior():
    """[M3.5-r2] 默认 15/×2 = M3 现行 ChunkPlanner 行为（growth 测试覆盖）。"""
    c = ServerConfig()
    assert c.codec_chunk_frames == 15
    assert c.codec_chunk_scale == 2
