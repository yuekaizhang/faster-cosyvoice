# tests/gpu/test_server_e2e.py
"""Start the server, run four streams plus one offline request, and validate audio.

Run with ``pytest tests/gpu/test_server_e2e.py -m gpu -v``; it takes about ten minutes.
"""
import asyncio
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest
import soundfile as sf

PORT = 18100
URL = f"http://localhost:{PORT}"
OUT = "results/pytest_server"

TEXTS = ["今天天气真不错，我们一起去公园散步吧。",
         "人工智能正在改变我们的生活方式。",
         "请记得明天早上八点开会。",
         "这本书的内容非常有趣，推荐大家阅读。"]


def _ref_data_url():
    from datasets import Audio, load_dataset
    ds = load_dataset("yuekai/seed_tts_cosy2", split="wenetspeech4tts")
    ds = ds.cast_column("prompt_audio", Audio(decode=False))
    row = ds[0]
    # Dataset bytes contain a complete WAV file and can be encoded directly.
    b64 = base64.b64encode(row["prompt_audio"]["bytes"]).decode()
    return "data:audio/wav;base64," + b64, row["prompt_text"]


async def _stream_one(client, ref_url, ref_text, text, idx):
    req = dict(input=text, ref_audio=ref_url, ref_text=ref_text,
               stream=True, response_format="wav", seed=100 + idx)
    t0 = time.perf_counter()
    ttfa = None
    data = b""
    async with client.stream("POST", f"{URL}/v1/audio/speech", json=req,
                             timeout=300) as r:
        assert r.status_code == 200
        async for chunk in r.aiter_bytes():
            if ttfa is None and len(data) + len(chunk) > 44:  # First audio byte.
                ttfa = time.perf_counter() - t0
            data += chunk
    pcm = np.frombuffer(data[44:], dtype="<i2").astype(np.float32) / 32767
    return ttfa or (time.perf_counter() - t0), pcm


@pytest.mark.gpu
def test_server_streaming_e2e():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    # FCV_E2E_SERVER_ARGS appends optional server flags for configuration tests.
    extra = shlex.split(os.environ.get("FCV_E2E_SERVER_ARGS", ""))
    proc = subprocess.Popen(
        [sys.executable, "-m", "faster_cosyvoice.server.app",
         "--port", str(PORT), *extra])
    try:
        # Engine loading is part of warmup, so readiness gets a generous timeout.
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server 提前退出 rc={proc.returncode}")
            try:
                if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("server 未就绪")

        ref_url, ref_text = _ref_data_url()

        async def run_all():
            async with httpx.AsyncClient() as client:
                return await asyncio.gather(*[
                    _stream_one(client, ref_url, ref_text, t, i)
                    for i, t in enumerate(TEXTS)])

        results = asyncio.run(run_all())
        expected = {}
        for i, (ttfa, pcm) in enumerate(results):
            assert len(pcm) > 24000 * 0.5, f"req{i} 音频过短"
            assert np.abs(pcm).mean() > 1e-4, f"req{i} 疑似静音"
            assert ttfa < 5.0, f"req{i} TTFA {ttfa:.1f}s 异常"
            sf.write(os.path.join(OUT, f"stream_{i}.wav"), pcm, 24000)
            expected[f"stream_{i}"] = TEXTS[i]
        print("TTFA(ms):", [round(t * 1000) for t, _ in results])

        # Generate the same seeded request through the non-streaming path.
        req = dict(input=TEXTS[0], ref_audio=ref_url, ref_text=ref_text,
                   stream=False, response_format="wav", seed=100)
        r = httpx.post(f"{URL}/v1/audio/speech", json=req, timeout=300)
        assert r.status_code == 200
        with open(os.path.join(OUT, "nonstream_0.wav"), "wb") as f:
            f.write(r.content)
        expected["nonstream_0"] = TEXTS[0]
        with open(os.path.join(OUT, "expected.json"), "w") as f:
            json.dump(expected, f, ensure_ascii=False)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
