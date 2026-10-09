import pytest

from faster_cosyvoice.llm.tokens import SpeechTokenCodec


class VocabTok:
    def get_vocab(self):
        v = {f"<|s_{i}|>": 1000 + i for i in range(6561)}
        v["<|eos1|>"] = 158486
        return v


def test_extract_and_eos():
    c = SpeechTokenCodec(VocabTok())
    assert c.eos_token_id == 158486
    # Non-speech tokens, including EOS, are filtered out.
    assert c.extract([1000, 158486, 1005, 42]) == [0, 5]


def test_missing_speech_token_raises():
    class Bad:
        def get_vocab(self):
            return {"<|eos1|>": 1}
    with pytest.raises(ValueError):
        SpeechTokenCodec(Bad())


def test_missing_eos_raises():
    class NoEos:
        def get_vocab(self):
            return {f"<|s_{i}|>": 1000 + i for i in range(6561)}
    with pytest.raises(ValueError):
        SpeechTokenCodec(NoEos())
