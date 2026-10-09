"""CosyVoice3 Flow-to-Mel and HiFT Mel-to-wave inference pipeline.

``CosyVoice3Token2Wav`` supports offline batches, individual streaming steps,
and packed cross-session streaming.  Request-specific mutable state stays in
``StreamSession`` so one model instance can safely interleave many requests.
"""
from typing import Optional

import torch

from faster_cosyvoice.token2wav.builders import build_flow, build_hift

# Offline HiFT compilation uses 64-frame (1.28-second) buckets.  Startup warms
# every bucket through 1280 frames (25.6 seconds); longer shapes build lazily.
_HIFT_BUCKET = 64
_HIFT_WARMUP_MAX = 1280


def _enable_hift_compile(hift, device: str) -> None:
    """Compile HiFT decode and bucket final-chunk Mel lengths.

    Padding collapses many dynamic lengths onto warm cuDNN plans.  The wrapper
    slices the waveform back to its unpadded length.  Non-final streaming chunks
    bypass padding because their right-lookahead region contains real context.
    Initialization compiles that path once and warms the common final buckets.
    """
    eager_decode = hift.decode
    compiled = torch.compile(eager_decode, dynamic=True)
    # One 50 Hz Mel frame maps to 480 samples at 24 kHz.
    import numpy as np
    ratio = int(np.prod(hift.upsample_rates)) * hift.istft_params["hop_len"]

    def decode(x, s=None, finalize=True):
        if s is None:
            s = torch.zeros(1, 1, 0)
        if not finalize or s.shape[2] == 0:
            return compiled(x=x, s=s, finalize=finalize)
        t = x.shape[2]
        pad = (-t) % _HIFT_BUCKET
        if pad == 0:
            return compiled(x=x, s=s, finalize=True)
        x = torch.nn.functional.pad(x, (0, pad))
        s = torch.nn.functional.pad(s, (0, pad * ratio))
        return compiled(x=x, s=s, finalize=True)[:, :t * ratio]

    hift.decode = decode
    # Inductor and Triton launch on the current device, which must match HiFT.
    with torch.inference_mode(), torch.cuda.device(device):
        for t in range(_HIFT_BUCKET, _HIFT_WARMUP_MAX + 1, _HIFT_BUCKET):
            mel = torch.zeros(1, 80, t, device=device) - 6.0
            hift.inference(speech_feat=mel, finalize=True)
        # Compile the non-final streaming branch once without padding.
        hift.inference(speech_feat=torch.zeros(1, 80, 200, device=device) - 6.0,
                       finalize=False)


class CosyVoice3Token2Wav(torch.nn.Module):
    def __init__(self, model_dir: str, device: str = "cuda:0",
                 estimator_mode: str = "flashinfer",
                 cuda_graph_buckets: Optional[list] = None,
                 hift_compile: bool = False,
                 stream_graph_buckets: Optional[list] = None,
                 hift_graph_buckets: Optional[list] = None):
        super().__init__()
        self.device = device
        self.fp16 = False
        self.flow = build_flow()
        self.flow.load_state_dict(
            torch.load(f"{model_dir}/flow.pt", map_location="cpu",
                       weights_only=True), strict=True)
        self.flow.to(device).eval()
        self.hift = build_hift()
        hift_sd = {k.replace("generator.", ""): v for k, v in torch.load(
            f"{model_dir}/hift.pt", map_location="cpu",
            weights_only=True).items()}
        self.hift.load_state_dict(hift_sd, strict=True)
        self.hift.to(device).eval()
        # Graph capture needs the eager decoder even when torch.compile wraps it.
        eager_hift_decode = self.hift.decode
        if hift_compile:
            _enable_hift_compile(self.hift, device)
        if hift_graph_buckets:
            # Non-final chunks capture lazily per bucket.  Final and overlong
            # chunks retain the original inference path.
            from faster_cosyvoice.token2wav.hift_graph import HiftStreamGraph
            HiftStreamGraph(self.hift, device,
                            list(hift_graph_buckets),
                            eager_decode=eager_hift_decode).install()

        assert estimator_mode in ("flashinfer", "torch"), estimator_mode
        self.estimator_mode = estimator_mode
        if estimator_mode == "flashinfer":
            from faster_cosyvoice.token2wav.flashinfer_dit import apply_flashinfer
            # CUDA Graphs apply only to one logical request (two CFG rows).
            # Multi-request batches continue through packed ragged attention.
            apply_flashinfer(
                self,
                enable_cuda_graph=bool(cuda_graph_buckets),
                cuda_graph_buckets=cuda_graph_buckets,
                stream_graph_buckets=stream_graph_buckets,
            )  # Replaces the estimator and moves Flow inference to fp16.

    @torch.inference_mode()
    def offline_batch(self, generated_tokens_list: list, conds: list,
                      max_batch: int = 8) -> list:
        """Convert complete speech-token sequences to 24 kHz waveforms."""
        wavs = []
        for s in range(0, len(generated_tokens_list), max_batch):
            wavs.extend(self._batch_impl(
                generated_tokens_list[s:s + max_batch], conds[s:s + max_batch]))
        return wavs

    def _batch_impl(self, tokens_list, conds):
        # FlashInfer and Triton use the thread-local current CUDA device.
        with torch.cuda.device(self.device):
            return self._batch_impl_pinned(tokens_list, conds)

    def _batch_impl_pinned(self, tokens_list, conds):
        if self.estimator_mode == "flashinfer":
            from faster_cosyvoice.token2wav.flashinfer_dit import flow_inference_batched
            token_list = [c.prompt_tokens_flow + list(t)
                          for c, t in zip(conds, tokens_list, strict=True)]
            prompt_feat_list = [c.prompt_feat.to(self.device) for c in conds]
            emb = torch.stack([c.spk_embedding for c in conds]).to(self.device)
            mels = flow_inference_batched(
                self.flow, token_list, prompt_feat_list, emb)
        else:
            mels = [self._flow_single(t, c)
                    for t, c in zip(tokens_list, conds, strict=True)]
        return [self.hift.inference(speech_feat=mel, finalize=True)[0].cpu()
                for mel in mels]

    def _flow_single(self, tokens, cond):
        """Run the original Torch Flow path for one request."""
        token = torch.tensor([list(tokens)], device=self.device)
        prompt_token = torch.tensor([cond.prompt_tokens_flow], device=self.device)
        prompt_feat = cond.prompt_feat.to(self.device)
        embedding = cond.spk_embedding.unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(self.fp16):
            mel, _ = self.flow.inference(
                token=token,
                token_len=torch.tensor([token.shape[1]], device=self.device),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                              device=self.device),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]],
                                             device=self.device),
                embedding=embedding,
                streaming=False, finalize=True)
        return mel

    @torch.inference_mode()
    def stream_step(self, session, plan) -> torch.Tensor:
        """Generate one streaming audio chunk for a session.

        Flow recomputes the available token prefix.  Only new Mel frames are
        appended to the session cache, then HiFT recomputes the cached Mel and
        returns audio after ``speech_offset``.  The result is CPU fp32 ``(1, N)``.
        """
        assert self.estimator_mode in ("torch", "flashinfer"), \
            self.estimator_mode
        with torch.cuda.device(self.device):
            return self._stream_step_pinned(session, plan)

    def _stream_step_pinned(self, session, plan):
        cond = session.cond
        token = torch.tensor([session.tokens[:plan.prefix_len]],
                             device=self.device)
        prompt_token = torch.tensor([cond.prompt_tokens_flow],
                                    device=self.device)
        prompt_feat = cond.prompt_feat.to(self.device)
        embedding = cond.spk_embedding.unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=self.fp16):
            mel, _ = self.flow.inference(
                token=token,
                token_len=torch.tensor([token.shape[1]], device=self.device),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                              device=self.device),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]],
                                             device=self.device),
                embedding=embedding,
                streaming=True, finalize=plan.finalize)
        return self._finish_chunk(session, plan, mel)

    def _finish_chunk(self, session, plan, mel_full: torch.Tensor
                      ) -> torch.Tensor:
        """Append new Mel frames, vocode the cache, and return only new PCM."""
        mel = mel_full[:, :, plan.token_offset * self.flow.token_mel_ratio:]
        if session.mel_cache is not None:
            mel = torch.cat([session.mel_cache, mel], dim=2)
        session.mel_cache = mel
        speech, _ = self.hift.inference(speech_feat=mel,
                                        finalize=plan.finalize)
        new = speech[:, session.speech_offset:]
        session.speech_offset += new.shape[1]
        session.chunk_index += 1
        return new.cpu()

    @torch.inference_mode()
    def stream_step_batched(self, sessions, plans) -> list:
        """Run one packed FlashInfer Flow step across independent sessions.

        Flow is batched with a per-document chunk-causal mask.  Vocoding then
        finishes each session independently in input order.
        """
        assert self.estimator_mode == "flashinfer", (
            "stream_step_batched requires estimator_mode='flashinfer', got "
            f"{self.estimator_mode}")
        assert len(sessions) == len(plans) and len(sessions) > 0
        from faster_cosyvoice.token2wav.flashinfer_dit import flow_inference_batched_streaming
        token_list = [list(s.cond.prompt_tokens_flow)
                      + list(s.tokens[:p.prefix_len])
                      for s, p in zip(sessions, plans, strict=True)]
        prompt_feat_list = [s.cond.prompt_feat.to(self.device)
                            for s in sessions]
        emb = torch.stack([s.cond.spk_embedding
                           for s in sessions]).to(self.device)
        finalize_list = [p.finalize for p in plans]
        with torch.cuda.device(self.device):
            mels = flow_inference_batched_streaming(
                self.flow, token_list, prompt_feat_list, emb, finalize_list)
            return [self._finish_chunk(s, p, mel)
                    for s, p, mel in zip(sessions, plans, mels, strict=True)]
