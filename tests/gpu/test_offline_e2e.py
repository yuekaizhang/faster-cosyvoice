# tests/gpu/test_offline_e2e.py
"""Run offline inference end to end and validate the generated audio."""
import json
import os
import shutil
import subprocess
import sys

import pytest
import soundfile as sf

OUT = "results/pytest_e2e"


@pytest.mark.gpu
def test_offline_e2e_dataset():
    cmd = [sys.executable, "examples/offline_inference.py",
           "--dataset", "yuekai/seed_tts_cosy2", "--split", "wenetspeech4tts",
           "--limit", "4", "--batch-size", "2",
           "--output-dir", OUT]
    shutil.rmtree(OUT, ignore_errors=True)
    subprocess.run(cmd, check=True, timeout=1800)
    metrics = json.load(open(os.path.join(OUT, "metrics.json")))
    assert metrics["failed"] == []
    assert metrics["finished_by_stop"] >= 3  # Allow an occasional length truncation.
    wavs = [f for f in os.listdir(OUT) if f.endswith(".wav")]
    assert len(wavs) == 4
    for f in wavs:
        audio, sr = sf.read(os.path.join(OUT, f), dtype="float32")
        assert sr == 24000
        assert abs(audio).mean() > 1e-4, f"{f} 疑似静音"
        assert len(audio) > sr * 0.5, f"{f} 过短"
