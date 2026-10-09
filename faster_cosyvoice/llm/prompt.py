"""Build CosyVoice3 voice-cloning prompts compatible with SpeechSpec."""

# The asymmetric quote set intentionally matches SpeechSpec byte for byte.
PUNCTS = ['"', "(", ")", "“", "”", "‘", "（", "）", "'"]
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
