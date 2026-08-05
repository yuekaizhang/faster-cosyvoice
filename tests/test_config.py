# tests/test_config.py
from faster_cosyvoice.config import LLMConfig, Token2WavConfig


def test_llm_defaults_match_spec_appendix_a():
    c = LLMConfig()
    assert c.target_model == "yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF"
    assert c.draft_model == "yuekai/cosyvoice3_llm_dspark"
    assert (c.temperature, c.top_p, c.top_k) == (0.8, 0.95, 15)
    assert c.repetition_penalty == 1.1
    assert c.max_tokens == 2048
    assert c.gpu_memory_utilization == 0.8


def test_token2wav_defaults():
    c = Token2WavConfig()
    assert c.estimator_mode == "flashinfer"
    assert c.batch_size == 8
