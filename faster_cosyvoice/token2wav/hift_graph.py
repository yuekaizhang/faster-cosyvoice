"""Bucketed CUDA Graph replay for non-final streaming HiFT chunks.

Each bucket captures one complete HiFT inference and turns its many kernel
launches into one replay.  Inputs are right-padded to the bucket, and outputs
are sliced back to the length produced by the unpadded eager call.  Non-final
HiFT output length is linear in Mel length; initialization fits that relation
with two probes and validates it with a third.

Final chunks stay eager because their boundary behavior differs from padded
intermediate chunks.  Captured outputs are views of static graph buffers and
must be consumed before the next replay; the single Token2Wav worker provides
that serialization.
"""
import contextlib
import logging

import torch

logger = logging.getLogger(__name__)

_OUTPUT_LENGTH_PROBES = (64, 96, 160)


def _istft_capture_safe(hift, magnitude, phase):
    """Capture-safe equivalent of HiFT's centered ``torch.istft`` call.

    ATen synchronizes the device while checking the window envelope, which
    invalidates CUDA Graph capture.  This implementation uses ``irfft`` and
    overlap-add and is installed only while a graph is captured.
    """
    n_fft = hift.istft_params["n_fft"]
    hop = hift.istft_params["hop_len"]
    window = hift.stft_window
    magnitude = torch.clip(magnitude, max=1e2)
    spec = torch.complex(magnitude * torch.cos(phase),
                         magnitude * torch.sin(phase))  # (B, F, T)
    frames = torch.fft.irfft(spec, n=n_fft, dim=1)      # (B, n_fft, T)
    frames = frames * window.view(1, -1, 1)
    b, _, t = frames.shape
    length = n_fft + hop * (t - 1)
    y = torch.nn.functional.fold(
        frames, output_size=(1, length), kernel_size=(1, n_fft),
        stride=(1, hop)).reshape(b, length)
    env = torch.nn.functional.fold(
        (window * window).view(1, -1, 1).expand(1, n_fft, t),
        output_size=(1, length), kernel_size=(1, n_fft),
        stride=(1, hop)).reshape(length)
    half = n_fft // 2
    return y[:, half:length - half] / env[half:length - half]


def _move_static_buffers(hift, device) -> None:
    """Move fixed noise and STFT buffers to the GPU before graph capture."""
    hift.stft_window = hift.stft_window.to(device)
    ms = hift.m_source
    if getattr(ms, "uv", None) is not None:
        ms.uv = ms.uv.to(device)
    sg = ms.l_sin_gen
    if getattr(sg, "rand_ini", None) is not None:
        sg.rand_ini = sg.rand_ini.to(device)
    if getattr(sg, "sine_waves", None) is not None:
        sg.sine_waves = sg.sine_waves.to(device)


class HiftStreamGraph:
    """Install lazy per-bucket graphs around ``hift.inference``.

    Eligible non-final calls use a graph when their Mel length fits the largest
    bucket.  Final or overlong calls fall back to the original implementation.
    """

    def __init__(self, hift, device: str, buckets: list[int],
                 eager_decode=None):
        self.hift = hift
        self.device = torch.device(device)
        self.buckets = sorted(int(b) for b in buckets)
        self.eager_decode = eager_decode
        self.pool = torch.cuda.graph_pool_handle()
        self.entries = {}
        self.orig_inference = hift.inference
        _move_static_buffers(hift, self.device)
        with torch.cuda.device(self.device):
            self._fit_out_len()

    def _eager_infer(self, t: int) -> torch.Tensor:
        mel = torch.zeros(1, 80, t, device=self.device) - 6.0
        with self._eager_decode_ctx():
            return self.orig_inference(speech_feat=mel, finalize=False)[0]

    def _fit_out_len(self) -> None:
        t0, t1, t2 = _OUTPUT_LENGTH_PROBES
        l0 = self._eager_infer(t0).shape[1]
        l1 = self._eager_infer(t1).shape[1]
        slope = (l1 - l0) // (t1 - t0)
        self._slope = slope
        self._intercept = l0 - slope * t0
        l2 = self._eager_infer(t2).shape[1]
        assert l2 == self.out_len(t2), (
            f"HiFT finalize=False output length is not linear: L({t2})={l2} "
            f"!= {self.out_len(t2)}")

    def out_len(self, t: int) -> int:
        return self._slope * t + self._intercept

    @contextlib.contextmanager
    def _eager_decode_ctx(self, capture_safe_istft: bool = False):
        """Temporarily restore eager decode and optionally capture-safe ISTFT."""
        current_decode = self.hift.decode
        current_istft = self.hift._istft
        if self.eager_decode is not None:
            self.hift.decode = self.eager_decode
        if capture_safe_istft:
            self.hift._istft = lambda m, p: _istft_capture_safe(self.hift, m, p)
        try:
            yield
        finally:
            self.hift.decode = current_decode
            self.hift._istft = current_istft

    def _capture(self, bucket: int) -> dict:
        with torch.cuda.device(self.device):
            return self._capture_impl(bucket)

    def _capture_impl(self, bucket: int) -> dict:
        static_in = torch.zeros(1, 80, bucket, device=self.device)
        with self._eager_decode_ctx(capture_safe_istft=True):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                # Warm exact-shape cuDNN/cuFFT plans and workspaces.
                for _ in range(2):
                    self.orig_inference(speech_feat=static_in, finalize=False)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out, _ = self.orig_inference(speech_feat=static_in,
                                             finalize=False)
        logger.info("hift stream graph captured: bucket=%d out=%s",
                    bucket, tuple(out.shape))
        return {"graph": graph, "in": static_in, "out": out}

    def run(self, speech_feat: torch.Tensor):
        """Pad, replay, and return a correctly sized static-buffer view."""
        t = speech_feat.shape[2]
        bucket = next(b for b in self.buckets if b >= t)
        entry = self.entries.get(bucket)
        if entry is None:
            entry = self._capture(bucket)
            self.entries[bucket] = entry
        s = entry["in"]
        s[:, :, :t].copy_(speech_feat)
        s[:, :, t:].zero_()
        entry["graph"].replay()
        return entry["out"][:, :self.out_len(t)], None

    def install(self) -> None:
        orig = self.orig_inference
        buckets_max = self.buckets[-1]

        def inference(speech_feat: torch.Tensor, finalize: bool = True):
            if (not finalize and speech_feat.shape[0] == 1
                    and speech_feat.shape[2] <= buckets_max):
                with torch.cuda.device(self.device):
                    return self.run(speech_feat)
            return orig(speech_feat=speech_feat, finalize=finalize)

        self.hift.inference = inference
        self.hift._stream_graph = self  # Expose the graph for tests and diagnostics.
