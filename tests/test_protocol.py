"""Server defaults, request validation, and reference-audio decoding."""
import base64
import io

import numpy as np
import pytest
import soundfile as sf

from faster_cosyvoice.config import ServerConfig
from faster_cosyvoice.server.protocol import SpeechRequest, VoiceRequest, decode_ref_audio


def _wav_data_url(seconds=1.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr,
             format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def test_server_config_defaults():
    c = ServerConfig()
    assert c.gpu_memory_utilization == 0.5
    assert c.max_ref_seconds == 30
    assert c.port == 8000


def test_speech_request_validation():
    r = SpeechRequest(input="你好", ref_audio=_wav_data_url(), ref_text="嗯")
    assert r.response_format == "wav" and r.stream is False
    with pytest.raises(ValueError):
        SpeechRequest(input="", ref_audio="x", ref_text="y")  # Empty input.
    with pytest.raises(ValueError):
        SpeechRequest(input="你好", response_format="mp3",     # Unsupported format.
                      ref_audio="x", ref_text="y")


def test_speech_request_accepts_nari_benchmark_fields():
    request = SpeechRequest(
        model="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF",
        input="Hello from the benchmark.",
        voice="benchmark",
        language="English",
        stream=True,
        non_streaming_mode=False,
        response_format="pcm",
    )
    assert request.language == "English"
    assert request.non_streaming_mode is False


def test_decode_data_url_and_truncate():
    wav, sr = decode_ref_audio(_wav_data_url(seconds=2.0), max_seconds=1.0)
    assert sr == 16000 and abs(len(wav) - 16000) <= 1  # Truncated to one second.


def test_voice_or_ref_required():
    with pytest.raises(ValueError):
        SpeechRequest(input="你好")  # Neither a voice nor a reference pair.


def test_voice_request_name_required():
    with pytest.raises(ValueError):
        VoiceRequest(name="  ", ref_audio="x", ref_text="y")
