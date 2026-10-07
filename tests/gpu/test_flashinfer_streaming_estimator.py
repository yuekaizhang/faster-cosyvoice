# tests/gpu/test_flashinfer_streaming_estimator.py
"""[M3] Estimator-level parity gates for FlashInferDiT chunk-causal streaming.

Gate A (mask correctness, immune to fp16-vs-fp32 noise): with the chunk-causal
mask (chunk=50), outputs for a shorter sequence must be a stable prefix of the
longer sequence sharing the same inputs, up to floor(T_old/50)*50 frames; the
offline (full-attention) mask must NOT have this property.

Gate B (reference alignment): real flow.pt weights, torch DiT estimator (fp32,
streaming=True — the reference mask is built by the real DiT.forward streaming
branch, dit.py:163) vs apply_flashinfer'd FlashInferDiT (fp16 + custom mask).
rel diff < 0.15: the known-acceptable fp16 threshold from the duplex self_test
in flashinfer_dit.py (hidden activations at magnitude ~6000, fp16 ulp = 4, so
different summation orders legitimately diverge by a few percent).

Both gates run on the packed varlen path (default, triton available) AND the
plain eager path — forced by monkeypatching the module-level _HAS_TRITON flag,
which is what FlashInferDiT.forward branches on (clean, function-scoped, no
model rebuild needed since _fused_tail is False without cuda graphs anyway).

Run: pytest tests/gpu/test_flashinfer_streaming_estimator.py -m gpu -v (in container).
"""
import types

import pytest
import torch

pytestmark = pytest.mark.gpu

fdit = pytest.importorskip("faster_cosyvoice.token2wav.flashinfer_dit")

CHUNK = 50
DEVICE = "cuda:0"


@pytest.fixture(params=["packed", "eager"])
def attn_path(request, monkeypatch):
    """Force the plain eager path by disabling the module-level triton flag."""
    if request.param == "eager":
        monkeypatch.setattr(fdit, "_HAS_TRITON", False)
    return request.param


def _inputs(T, seed=1234, dtype=torch.float16):
    torch.manual_seed(seed)
    x = torch.randn(2, 80, T, device=DEVICE, dtype=dtype)
    mask = torch.ones(2, 1, T, device=DEVICE, dtype=dtype)
    mu = torch.randn(2, 80, T, device=DEVICE, dtype=dtype)
    t = torch.rand(1, device=DEVICE, dtype=dtype).expand(2).contiguous()
    spks = torch.randn(2, 80, device=DEVICE, dtype=dtype)
    cond = torch.randn(2, 80, T, device=DEVICE, dtype=dtype)
    return x, mask, mu, t, spks, cond


# ---------------------------------------------------------------- Gate A
@pytest.mark.gpu
@torch.inference_mode()
def test_gate_a_chunk_mask_prefix_stability(attn_path):
    torch.manual_seed(0)
    fi = fdit.FlashInferDiT(device=DEVICE)
    fi = fi.to(device=DEVICE, dtype=torch.float16).eval()
    fi.finalize_weights()

    T_old, T_new = 160, 200
    stable = (T_old // CHUNK) * CHUNK  # 150
    x, mask, mu, t, spks, cond = _inputs(T_new)

    def run(T, streaming):
        return fi(x[:, :, :T].contiguous(), mask[:, :, :T].contiguous(),
                  mu[:, :, :T].contiguous(), t, spks,
                  cond[:, :, :T].contiguous(), streaming=streaming)

    # streaming: first floor(160/50)*50 = 150 frames must be stable
    out_old_s = run(T_old, True)
    out_new_s = run(T_new, True)
    d_stream = (out_old_s[:, :, :stable]
                - out_new_s[:, :, :stable]).abs().max().item()
    assert torch.allclose(out_old_s[:, :, :stable], out_new_s[:, :, :stable],
                          atol=1e-3, rtol=0), \
        f"[{attn_path}] streaming prefix unstable: max abs diff {d_stream:.2e}"

    # offline (full attention): the same prefix must NOT be stable
    out_old_o = run(T_old, False)
    out_new_o = run(T_new, False)
    d_off = (out_old_o[:, :, :stable]
             - out_new_o[:, :, :stable]).abs().max().item()
    assert d_off > 5e-3, \
        f"[{attn_path}] offline prefix unexpectedly stable: {d_off:.2e}"
    print(f"[gate A/{attn_path}] streaming prefix diff {d_stream:.2e}, "
          f"offline prefix diff {d_off:.2e}")


# ---------------------------------------------------------------- Gate B
@pytest.fixture(scope="module")
def real_estimators():
    """torch DiT (fp32, reference) + FlashInferDiT via apply_flashinfer,
    both loaded from the real flow.pt."""
    from faster_cosyvoice.assets import ensure_token2wav_assets
    from faster_cosyvoice.config import Token2WavConfig
    from faster_cosyvoice.token2wav.builders import build_flow

    model_dir = ensure_token2wav_assets(Token2WavConfig().model_dir)
    sd = torch.load(f"{model_dir}/flow.pt", map_location="cpu",
                    weights_only=True)

    flow_ref = build_flow()
    flow_ref.load_state_dict(sd, strict=True)
    flow_ref.to(DEVICE).eval()

    flow_fi = build_flow()
    flow_fi.load_state_dict(sd, strict=True)
    flow_fi.to(DEVICE).eval()
    holder = types.SimpleNamespace(flow=flow_fi, fp16=False)
    fdit.apply_flashinfer(holder)  # halves flow_fi, swaps the estimator

    return flow_ref.decoder.estimator, holder.flow.decoder.estimator


@pytest.mark.gpu
@torch.inference_mode()
def test_gate_b_torch_streaming_reference(real_estimators, attn_path):
    ref, fi = real_estimators
    assert fi._chunk_size == ref.static_chunk_size == CHUNK

    T = 300
    x, mask, mu, t, spks, cond = _inputs(T, seed=7, dtype=torch.float32)

    # reference mask built by the real DiT.forward streaming branch
    # (add_optional_chunk_mask, dit.py:163) — not re-derived here.
    out_ref = ref(x, mask, mu, t, spks, cond, streaming=True)
    out_fi = fi(x, mask, mu, t, spks, cond, streaming=True).float()

    diff = (out_ref - out_fi).abs().max().item()
    rel = diff / out_ref.abs().max().item()
    print(f"[gate B/{attn_path}] T={T}: max_abs={diff:.5f} rel={rel:.5f}")
    # 0.15: duplex self_test fp16 convention (see flashinfer_dit.self_test)
    assert rel < 0.15, \
        f"[{attn_path}] streaming estimator diverges beyond fp16 noise: {rel}"
