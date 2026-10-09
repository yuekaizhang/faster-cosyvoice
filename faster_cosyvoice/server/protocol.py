"""Request models and reference-audio decoding for the speech API.

``ref_audio`` accepts a base64 data URI, an HTTP(S) URL, or a local path.
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
    voice: Optional[str] = None            # Registered server-side voice name.
    language: Optional[str] = None         # Compatibility field; CV3 detects it.
    ref_audio: Optional[str] = None        # data:/http(s)/path
    ref_text: Optional[str] = None
    response_format: str = "wav"           # wav | pcm
    # This controls HTTP delivery only.  Both modes use incremental synthesis.
    stream: bool = False
    # Accepted but ignored so Nari/vLLM-compatible clients need no special case.
    non_streaming_mode: bool = False
    seed: int = 42

    @field_validator("input")
    @classmethod
    def _non_empty(cls, v):
        if not v.strip():
            raise ValueError("input must not be empty")
        return v

    @field_validator("response_format")
    @classmethod
    def _fmt(cls, v):
        if v not in ("wav", "pcm"):
            raise ValueError("response_format must be 'wav' or 'pcm'")
        return v

    @model_validator(mode="after")
    def _voice_or_ref(self):
        if self.voice is None and not (self.ref_audio and self.ref_text):
            raise ValueError("provide voice or both ref_audio and ref_text")
        return self


class VoiceRequest(BaseModel):
    name: str
    ref_audio: str
    ref_text: str

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        if not v.strip():
            raise ValueError("name must not be empty")
        return v


def decode_ref_audio(ref: str, max_seconds: float = 30.0):
    """Decode mono float32 samples and truncate audio longer than the limit."""
    if ref.startswith("data:"):
        try:
            b64 = ref.split(",", 1)[1]
            data = io.BytesIO(base64.b64decode(b64, validate=True))
        except Exception as e:
            raise ValueError(f"could not parse ref_audio: {e}") from e
    elif ref.startswith(("http://", "https://")):
        import httpx
        try:
            resp = httpx.get(ref, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            raise ValueError(f"could not fetch ref_audio URL: {e}") from e
        data = io.BytesIO(resp.content)
    else:
        # Local paths and arbitrary URLs assume a trusted internal deployment.
        # Public deployments should disable this branch or add an allowlist.
        data = ref
    try:
        wav, sr = sf.read(data, dtype="float32")
    except Exception as e:
        raise ValueError(f"could not decode ref_audio: {e}") from e
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    limit = int(max_seconds * sr)
    if len(wav) > limit:
        logger.warning(
            "reference audio is %.1fs; truncating to %.0fs",
            len(wav) / sr,
            max_seconds,
        )
        wav = wav[:limit]
    return wav, sr
