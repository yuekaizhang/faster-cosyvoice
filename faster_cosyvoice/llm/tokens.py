"""vLLM 输出 token id → speech id 查表（不 detokenize；spec §5.1）。"""

SPEECH_VOCAB_SIZE = 6561  # CV3 flow vocab_size


class SpeechTokenCodec:
    def __init__(self, tokenizer, vocab_size: int = SPEECH_VOCAB_SIZE):
        vocab = tokenizer.get_vocab()
        self.id_to_speech: dict[int, int] = {}
        for n in range(vocab_size):
            tid = vocab.get(f"<|s_{n}|>")
            if tid is None:
                raise ValueError(
                    f"词表缺 <|s_{n}|>——target 模型不是 CV3 HF checkpoint？")
            self.id_to_speech[tid] = n
        if "<|eos1|>" not in vocab:
            raise ValueError("词表缺 <|eos1|>")
        self.eos_token_id: int = vocab["<|eos1|>"]

    def extract(self, token_ids) -> list[int]:
        m = self.id_to_speech
        return [m[t] for t in token_ids if t in m]
