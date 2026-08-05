"""CV3 voice-clone prompt 构建（迁自 SpeechSpec benchmark_tts.py，逐字节一致）。"""

PUNCTS = ['"', "(", ")", "“", "”", "‘", "（", "）", "'"]  # NOTE: 不对称（有'‘'无'’'）是有意的——与 SpeechSpec benchmark_tts.py 字节兼容，勿"修复"
COSYVOICE3_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def speech_id_str(tokens) -> str:
    return "".join(f"<|s_{t}|>" for t in tokens)


def build_prompt(tokenizer, ref_text: str, target_text: str,
                 prompt_speech_tokens) -> str:
    full_text = COSYVOICE3_PREFIX + ref_text + target_text
    for p in PUNCTS:
        full_text = full_text.replace(p, "")
    chat = [
        {"role": "user", "content": full_text},
        {"role": "assistant", "content": speech_id_str(prompt_speech_tokens)},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, continue_final_message=True)
