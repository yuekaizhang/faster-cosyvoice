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
    assert c.deadline_reserve_s == 0.1
    assert c.scheduler_mode == "deadline"
    assert c.offline_flow_graph_duration_buckets is None
    assert c.vocoder_compile is False
    assert c.streaming_flow_graph_buckets is None
    assert c.streaming_vocoder_graph_buckets is None
    assert c.speaker_encoder_tensorrt is False


def test_server_chunk_defaults_match_current_behavior():
    """[M3.5-r2] 默认 15/×2 = M3 现行 ChunkPlanner 行为（growth 测试覆盖）。"""
    c = ServerConfig()
    assert c.speech_token_chunk_size == 15
    assert c.speech_token_chunk_growth == 2
    assert c.trim_leading_silence is False
    assert c.leading_silence_preroll_ms == 20.0
    assert c.leading_silence_max_ms == 2000.0
    assert c.leading_silence_min_buffer_ms == 400.0
