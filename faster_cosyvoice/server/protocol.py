"""OpenAI /v1/audio/speech 请求模型 + ref 音频解析（spec §5.5/§7）。

ref_audio 支持：data:...;base64 URI、http(s) URL、本地路径。
"""
import base64
import io
import logging
from typing import Optional

import soundfile as sf
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    input: str
    model: str = "faster-cosyvoice"
    voice: Optional[str] = None            # 已注册音色名
    ref_audio: Optional[str] = None        # data:/http(s)/path
    ref_text: Optional[str] = None
    response_format: str = "wav"           # wav | pcm
    stream: bool = False
    seed: int = 42

    @field_validator("input")
    @classmethod
    def _non_empty(cls, v):
        if not v.strip():
            raise ValueError("input 不能为空")
        return v

    @field_validator("response_format")
    @classmethod
    def _fmt(cls, v):
        if v not in ("wav", "pcm"):
            raise ValueError("response_format 仅支持 wav|pcm（v1）")
        return v

    @model_validator(mode="after")
    def _voice_or_ref(self):
        if self.voice is None and not (self.ref_audio and self.ref_text):
            raise ValueError("需要 voice 或 (ref_audio + ref_text)")
        return self


class VoiceRequest(BaseModel):
    name: str
    ref_audio: str
    ref_text: str

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        if not v.strip():
            raise ValueError("name 不能为空")
        return v


def decode_ref_audio(ref: str, max_seconds: float = 30.0):
    """→ (1-D float32 numpy, sr)。超长截断并 warning（spec §7）。"""
    if ref.startswith("data:"):
        b64 = ref.split(",", 1)[1]
        data = io.BytesIO(base64.b64decode(b64))
    elif ref.startswith(("http://", "https://")):
        import httpx
        resp = httpx.get(ref, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        data = io.BytesIO(resp.content)
    else:
        data = ref  # 本地路径
    wav, sr = sf.read(data, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    limit = int(max_seconds * sr)
    if len(wav) > limit:
        logger.warning("ref 音频 %.1fs 超过 %.0fs，截断", len(wav) / sr,
                       max_seconds)
        wav = wav[:limit]
    return wav, sr
