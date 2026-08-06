# tests/test_engine.py
import json
import pytest
from faster_cosyvoice.config import LLMConfig
from faster_cosyvoice.llm.engine import build_llm_kwargs


def test_no_draft(tmp_path):
    kw = build_llm_kwargs(LLMConfig(target_model="m", draft_model=None))
    assert kw["model"] == "m"
    assert "speculative_config" not in kw
    assert kw["max_model_len"] == 4096


def test_draft_auto_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"speculators_model_type": "dspark", "block_size": 8}))
    cfg = LLMConfig(target_model="m", draft_model=str(tmp_path))
    kw = build_llm_kwargs(cfg)
    assert kw["speculative_config"] == {
        "model": str(tmp_path),
        "method": "dspark",
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
        "draft_apply_repetition_penalty": True,  # rp=1.1 默认 → mirror 开
    }


def test_draft_without_method_raises(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"block_size": 8}))
    with pytest.raises(ValueError):
        build_llm_kwargs(LLMConfig(target_model="m", draft_model=str(tmp_path)))


from faster_cosyvoice.llm.engine import make_stream_sampling_params


class FakeCodec:
    eos_token_id = 158486


def test_stream_sampling_params_dynamic_bounds():
    sp = make_stream_sampling_params(LLMConfig(), FakeCodec(),
                                     text_token_len=10, seed=7)
    assert sp.min_tokens == 20                      # 2×text
    assert sp.max_tokens == 200                     # min(2048, 20×text)
    assert sp.stop_token_ids == [158486]
    assert sp.detokenize is False
    assert sp.seed == 7


def test_stream_sampling_params_caps_at_2048():
    sp = make_stream_sampling_params(LLMConfig(), FakeCodec(),
                                     text_token_len=1000, seed=0)
    assert sp.max_tokens == 2048
