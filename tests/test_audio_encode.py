import struct

import torch

from faster_cosyvoice.server.audio_encode import pcm16_bytes, wav_stream_header


def test_wav_header_44_bytes_unknown_length():
    h = wav_stream_header(24000)
    assert len(h) == 44
    assert h[:4] == b"RIFF" and h[8:12] == b"WAVE"
    assert struct.unpack("<I", h[4:8])[0] == 0xFFFFFFFF  # unknown length
    assert struct.unpack("<I", h[24:28])[0] == 24000  # sample rate
    assert struct.unpack("<I", h[40:44])[0] == 0xFFFFFFFF  # data length


def test_pcm16_clip_and_dtype():
    wav = torch.tensor([[0.0, 1.0, -1.0, 2.0]])  # 2.0 gets clipped
    b = pcm16_bytes(wav)
    vals = struct.unpack("<4h", b)
    assert vals[0] == 0 and vals[1] == 32767
    assert vals[2] == -32767 and vals[3] == 32767
