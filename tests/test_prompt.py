import pytest

from faster_cosyvoice.llm.prompt import build_prompt, speech_id_str


class StubTokenizer:
    def apply_chat_template(self, chat, tokenize, continue_final_message):
        assert tokenize is False and continue_final_message is True
        return "|".join(f"{m['role']}:{m['content']}" for m in chat)


def test_speech_id_str():
    assert speech_id_str([0, 42]) == "<|s_0|><|s_42|>"


def test_build_prompt_prefix_puncts_and_structure():
    p = build_prompt(StubTokenizer(), ref_text="你好“引号”",
                     target_text="目标(括号)", prompt_speech_tokens=[1, 2])
    assert p == ("user:You are a helpful assistant.<|endofprompt|>你好引号目标括号"
                 "|assistant:<|s_1|><|s_2|>")


@pytest.mark.integration
def test_build_prompt_matches_speechspec_golden():
    """Prompt output remains byte-compatible with SpeechSpec."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF")
    p = build_prompt(tok, "你好。", "今天天气不错。", [10, 20, 30])
    assert p.endswith("<|s_10|><|s_20|><|s_30|>")
    assert "You are a helpful assistant.<|endofprompt|>你好。今天天气不错。" in p
