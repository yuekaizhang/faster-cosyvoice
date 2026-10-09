# tests/gpu/test_hift_graph.py
"""Validate bucketed streaming HiFT CUDA Graphs against eager inference.

Non-final graph calls must match shape and stay within atol=5e-3 across
buckets.  Final and overlong calls must fall back to bit-exact eager output.
Run with ``pytest tests/gpu/test_hift_graph.py -m gpu -v -s``.
"""
import pytest
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import Token2WavConfig
from faster_cosyvoice.token2wav.builders import build_hift
from faster_cosyvoice.token2wav.hift_graph import HiftStreamGraph

BUCKETS = [64, 128, 192, 256]


@pytest.fixture(scope="module")
def env():
    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    device = "cuda:0"
    hift = build_hift()
    sd = {k.replace("generator.", ""): v for k, v in torch.load(
        f"{model_dir}/hift.pt", map_location="cpu", weights_only=True).items()}
    hift.load_state_dict(sd, strict=True)
    hift.to(device).eval()
    eager_inference = hift.inference  # Save the original entry point.
    HiftStreamGraph(hift, device, BUCKETS).install()
    return hift, eager_inference, device


@pytest.mark.gpu
def test_gate_a_graph_matches_eager(env):
    hift, eager_inference, device = env
    torch.manual_seed(0)
    for t in (56, 106, 156, 206, 256):
        mel = torch.randn(1, 80, t, device=device) * 2 - 6.0
        ref, _ = eager_inference(speech_feat=mel.clone(), finalize=False)
        got, _ = hift.inference(speech_feat=mel.clone(), finalize=False)
        got = got.clone()  # Materialize the static-buffer view before replay.
        assert got.shape == ref.shape, (t, got.shape, ref.shape)
        assert torch.isfinite(got).all()
        diff = (got - ref).abs().max().item()
        assert torch.allclose(got, ref, atol=5e-3), (
            f"T={t}: max abs diff {diff:.3e}")
        print(f"T={t} out={ref.shape[1]} max_abs_diff={diff:.3e}")


@pytest.mark.gpu
def test_gate_b_fallback_paths(env):
    hift, eager_inference, device = env
    torch.manual_seed(1)
    # Final chunks use the original path and remain bit-exact.
    mel = torch.randn(1, 80, 100, device=device) * 2 - 6.0
    ref, _ = eager_inference(speech_feat=mel.clone(), finalize=True)
    got, _ = hift.inference(speech_feat=mel.clone(), finalize=True)
    assert torch.equal(got, ref)
    # Inputs beyond the largest bucket also use the original path.
    mel = torch.randn(1, 80, BUCKETS[-1] + 32, device=device) * 2 - 6.0
    ref, _ = eager_inference(speech_feat=mel.clone(), finalize=False)
    got, _ = hift.inference(speech_feat=mel.clone(), finalize=False)
    assert torch.equal(got, ref)
