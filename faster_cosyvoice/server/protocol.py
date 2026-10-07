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
    language: Optional[str] = None         # OpenAI/Nari 兼容字段；CV3 自动识别
    ref_audio: Optional[str] = None        # data:/http(s)/path
    ref_text: Optional[str] = None
    response_format: str = "wav"           # wav | pcm
    stream: bool = False
    non_streaming_mode: bool = False       # Nari/vLLM benchmark 兼容字段
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
        try:
            b64 = ref.split(",", 1)[1]
            data = io.BytesIO(base64.b64decode(b64, validate=True))
        except Exception as e:
            raise ValueError(f"ref_audio 无法解析: {e}") from e
    elif ref.startswith(("http://", "https://")):
        import httpx
        try:
            resp = httpx.get(ref, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            raise ValueError(f"ref_audio URL 拉取失败: {e}") from e
        data = io.BytesIO(resp.content)
    else:
        # 信任假设：内网部署，允许本地路径/任意 URL（外网部署需加白名单/关闭此分支）
        data = ref  # 本地路径
    try:
        wav, sr = sf.read(data, dtype="float32")
    except Exception as e:
        raise ValueError(f"ref_audio 无法解析: {e}") from e
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    limit = int(max_seconds * sr)
    if len(wav) > limit:
        logger.warning("ref 音频 %.1fs 超过 %.0fs，截断", len(wav) / sr,
                       max_seconds)
        wav = wav[:limit]
    return wav, sr
