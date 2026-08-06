# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-accelerated CosyVoice3 DiT flow-matching estimator.

Drop-in replacement for cosyvoice.flow.DiT.dit.DiT (offline path): state-dict
compatible, loads flow.pt weights unchanged, swaps in as
flow.decoder.estimator (the nn.Module branch of forward_estimator).

Optimizations:
- attention: SDPA + (b,n,n) materialized mask -> flashinfer ragged prefill
  (2 CFG documents, no mask); the partial x_transformers RoPE (first 64 of
  1024 channels) is exactly "rotate head 0 only" in NHD layout.
- adaLN: the modulate "+1" folded into the adaLN linear bias; every
  modulate / gated-residual collapses to one addcmul.
- qkv fused into one GEMM.
- optional CUDA graphs: per-shape, or duration-bucketed (both CFG docs share
  one length, so bucketed attention is a single SDPA call with a runtime
  key-padding mask).

Usage:
    from token2wav_cosyvoice3_flashinfer import apply_flashinfer
    model = CosyVoice3_Token2Wav(model_dir, enable_trt=False)
    apply_flashinfer(model, enable_cuda_graph=True)
"""
import math
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer

from x_transformers.x_transformers import RotaryEmbedding
from faster_cosyvoice.token2wav.cosyvoice.flow.DiT.dit import InputEmbedding
from faster_cosyvoice.token2wav.cosyvoice.flow.DiT.modules import (
    TimestepEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
)

_WORKSPACE_SIZE = 64 * 1024 * 1024
_ROPE_MAX_LEN = 4096

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ln_modulate_kernel(X, SHIFT, SCALE, OUT, n_per_batch, mod_stride,
                            eps: tl.constexpr, D: tl.constexpr):
        """out = LayerNorm(x, no affine) * scale + shift, one row per program.
        X/OUT: (rows, D) contiguous; SHIFT/SCALE: strided views, one row per
        batch element (row // n_per_batch), inner dim contiguous."""
        row = tl.program_id(0)
        cols = tl.arange(0, D)
        x = tl.load(X + row * D + cols).to(tl.float32)
        mean = tl.sum(x) / D
        xc = x - mean
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc) / D + eps)
        y = xc * rstd
        b = row // n_per_batch
        sh = tl.load(SHIFT + b * mod_stride + cols).to(tl.float32)
        sc = tl.load(SCALE + b * mod_stride + cols).to(tl.float32)
        tl.store(OUT + row * D + cols, (y * sc + sh).to(OUT.dtype.element_ty))

    @triton.jit
    def _qkv_rope_repack_kernel(QKV, Q, K, V, COS, SIN, n_per_batch,
                                D: tl.constexpr):
        """One kernel per row: split the fused-qkv GEMM output into packed
        q/k/v (replacing three .contiguous() copies) and rotate head 0 of
        q/k in place (x_transformers partial rope, interleaved pairs)."""
        row = tl.program_id(0)
        cols = tl.arange(0, D)
        base = row * 3 * D
        q = tl.load(QKV + base + cols).to(tl.float32)
        k = tl.load(QKV + base + D + cols).to(tl.float32)
        v = tl.load(QKV + base + 2 * D + cols)

        is_h0 = cols < 64
        pos = row % n_per_batch
        pair = cols // 2
        cos = tl.load(COS + pos * 32 + pair, mask=is_h0, other=1.0)
        sin = tl.load(SIN + pos * 32 + pair, mask=is_h0, other=0.0)
        partner = tl.where(cols % 2 == 0, cols + 1, cols - 1)
        sign = tl.where(cols % 2 == 0, -1.0, 1.0)
        qp = tl.load(QKV + base + partner, mask=is_h0, other=0.0).to(tl.float32)
        kp = tl.load(QKV + base + D + partner, mask=is_h0, other=0.0).to(tl.float32)
        q = tl.where(is_h0, q * cos + sign * qp * sin, q)
        k = tl.where(is_h0, k * cos + sign * kp * sin, k)

        tl.store(Q + row * D + cols, q.to(Q.dtype.element_ty))
        tl.store(K + row * D + cols, k.to(K.dtype.element_ty))
        tl.store(V + row * D + cols, v)

    @triton.jit
    def _gate_ln_modulate_kernel(H, GATE, Y, SHIFT, SCALE, H_OUT, NORM_OUT,
                                 n_per_batch, mod_stride,
                                 eps: tl.constexpr, D: tl.constexpr):
        """h_out = h + gate * y; norm_out = LayerNorm(h_out) * scale + shift.
        Fuses the gated residual into the next norm+modulate (two stores)."""
        row = tl.program_id(0)
        cols = tl.arange(0, D)
        b = row // n_per_batch
        h = tl.load(H + row * D + cols).to(tl.float32)
        g = tl.load(GATE + b * mod_stride + cols).to(tl.float32)
        y = tl.load(Y + row * D + cols).to(tl.float32)
        h = h + g * y
        tl.store(H_OUT + row * D + cols, h.to(H_OUT.dtype.element_ty))
        mean = tl.sum(h) / D
        hc = h - mean
        rstd = 1.0 / tl.sqrt(tl.sum(hc * hc) / D + eps)
        sh = tl.load(SHIFT + b * mod_stride + cols).to(tl.float32)
        sc = tl.load(SCALE + b * mod_stride + cols).to(tl.float32)
        tl.store(NORM_OUT + row * D + cols,
                 (hc * rstd * sc + sh).to(NORM_OUT.dtype.element_ty))

    @triton.jit
    def _ln_modulate_packed_kernel(X, SHIFT, SCALE, OUT, DOC, mod_stride,
                                   eps: tl.constexpr, D: tl.constexpr):
        """Packed-layout variant: per-row document id instead of row//n."""
        row = tl.program_id(0)
        cols = tl.arange(0, D)
        x = tl.load(X + row * D + cols).to(tl.float32)
        mean = tl.sum(x) / D
        xc = x - mean
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc) / D + eps)
        b = tl.load(DOC + row)
        sh = tl.load(SHIFT + b * mod_stride + cols).to(tl.float32)
        sc = tl.load(SCALE + b * mod_stride + cols).to(tl.float32)
        tl.store(OUT + row * D + cols,
                 (xc * rstd * sc + sh).to(OUT.dtype.element_ty))

    @triton.jit
    def _gate_ln_modulate_packed_kernel(H, GATE, Y, SHIFT, SCALE, H_OUT,
                                        NORM_OUT, DOC, mod_stride,
                                        eps: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, D)
        b = tl.load(DOC + row)
        h = tl.load(H + row * D + cols).to(tl.float32)
        g = tl.load(GATE + b * mod_stride + cols).to(tl.float32)
        y = tl.load(Y + row * D + cols).to(tl.float32)
        h = h + g * y
        tl.store(H_OUT + row * D + cols, h.to(H_OUT.dtype.element_ty))
        mean = tl.sum(h) / D
        hc = h - mean
        rstd = 1.0 / tl.sqrt(tl.sum(hc * hc) / D + eps)
        sh = tl.load(SHIFT + b * mod_stride + cols).to(tl.float32)
        sc = tl.load(SCALE + b * mod_stride + cols).to(tl.float32)
        tl.store(NORM_OUT + row * D + cols,
                 (hc * rstd * sc + sh).to(NORM_OUT.dtype.element_ty))

    @triton.jit
    def _qkv_rope_repack_packed_kernel(QKV, Q, K, V, COS, SIN, POS,
                                       D: tl.constexpr):
        """Packed variant: per-row rope position from POS (restarts per doc)."""
        row = tl.program_id(0)
        cols = tl.arange(0, D)
        base = row * 3 * D
        q = tl.load(QKV + base + cols).to(tl.float32)
        k = tl.load(QKV + base + D + cols).to(tl.float32)
        v = tl.load(QKV + base + 2 * D + cols)

        is_h0 = cols < 64
        pos = tl.load(POS + row)
        pair = cols // 2
        cos = tl.load(COS + pos * 32 + pair, mask=is_h0, other=1.0)
        sin = tl.load(SIN + pos * 32 + pair, mask=is_h0, other=0.0)
        partner = tl.where(cols % 2 == 0, cols + 1, cols - 1)
        sign = tl.where(cols % 2 == 0, -1.0, 1.0)
        qp = tl.load(QKV + base + partner, mask=is_h0, other=0.0).to(tl.float32)
        kp = tl.load(QKV + base + D + partner, mask=is_h0, other=0.0).to(tl.float32)
        q = tl.where(is_h0, q * cos + sign * qp * sin, q)
        k = tl.where(is_h0, k * cos + sign * kp * sin, k)

        tl.store(Q + row * D + cols, q.to(Q.dtype.element_ty))
        tl.store(K + row * D + cols, k.to(K.dtype.element_ty))
        tl.store(V + row * D + cols, v)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _ln_modulate_triton(x, shift, scale, eps=1e-6):
    """Fused LayerNorm(elementwise_affine=False) + modulate (one kernel).
    Saves kernels + memory passes, but triton's Python launcher costs
    30-60us of host time per call — a win only inside CUDA graphs.
    x: (b, n, D) contiguous; shift/scale: (b, 1, D) views with contiguous
    inner dim (slices of the fused adaLN output)."""
    b, n, d = x.shape
    out = torch.empty_like(x)
    _ln_modulate_kernel[(b * n,)](
        x, shift, scale, out, n, shift.stride(0), eps, d)
    return out


def _ln_modulate_torch(x, shift, scale, eps=1e-6):
    return torch.addcmul(shift, F.layer_norm(x, (x.shape[-1],), eps=eps), scale)


def _gate_ln_modulate(h, gate, y, shift, scale, eps=1e-6):
    """h_out = h + gate*y; norm_out = LN(h_out)*scale + shift (one kernel)."""
    b, n, d = h.shape
    h_out = torch.empty_like(h)
    norm_out = torch.empty_like(h)
    _gate_ln_modulate_kernel[(b * n,)](
        h, gate, y, shift, scale, h_out, norm_out, n, gate.stride(0), eps, d)
    return h_out, norm_out


def _chunk_causal_flat_mask(doc_lens, chunk_size: int, device) -> torch.Tensor:
    """[M3] per-doc row-major chunk-causal bool mask, flattened and
    concatenated in doc order (flashinfer custom_mask layout, True=keep).
    Predicate allowed(q,k) = k//chunk <= q//chunk, equivalent to the vendored
    subsequent_chunk_mask (cf. dit.py streaming branch, dit.py:163)."""
    parts = []
    for n in doc_lens:
        idx = torch.arange(n, device=device)
        parts.append((idx.view(1, -1) // chunk_size
                      <= idx.view(-1, 1) // chunk_size).flatten())
    return torch.cat(parts)


class RaggedAttentionRunner:
    def __init__(self, num_heads, head_dim, device, workspace_size=_WORKSPACE_SIZE):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self._workspace = torch.zeros(workspace_size, dtype=torch.uint8, device=device)
        self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace, "NHD"
        )
        self._planned_key = None
        # [M3.5] each runner serves ONE plan key at a time (the flashinfer
        # wrapper holds a single plan internally), so its mask cache stays
        # single-slot; multi-key reuse is handled one level up by the
        # FlashInferDiT runner pool (see _planned_runner), which keeps whole
        # ready-planned runners alive instead of caching masks alone —
        # profiling showed the mask build is only ~2.4ms of the ~17ms
        # chunk-1 plan_docs; the other ~14ms is wrapper.plan() itself.
        self._mask_key = None
        self._mask = None
        self.plan_calls = 0  # [M3.5] actual (non-no-op) plans; test/profiling hook

    def _custom_mask(self, doc_lens, chunk_size):
        """[M3] single-slot cached flat chunk-causal mask (saves ~1.3ms
        rebuild across the 10 euler steps; old mask freed by refcount).
        NOTE: we pass this as custom_mask, NEVER packed_custom_mask —
        flashinfer 0.6.13's packed_custom_mask path has a byte-vs-element
        unit mismatch for multi-doc ragged batches (wrong mask applied)."""
        key = (tuple(doc_lens), chunk_size)
        if key != self._mask_key:
            if len(set(doc_lens)) == 1:
                # equal-length docs (e.g. the b=2 CFG batch): build one
                # T x T block and tile it instead of B identical blocks
                block = _chunk_causal_flat_mask(
                    doc_lens[:1], chunk_size, self.device)
                self._mask = block.repeat(len(doc_lens))
            else:
                self._mask = _chunk_causal_flat_mask(
                    doc_lens, chunk_size, self.device)
            self._mask_key = key
        return self._mask

    def plan(self, batch_size: int, seq_len: int, dtype: torch.dtype,
             chunk_size: Optional[int] = None):  # [M3] chunk-causal streaming
        key = (batch_size, seq_len, dtype, chunk_size)  # [M3] key incl. chunk
        if key == self._planned_key:
            return
        self.plan_calls += 1  # [M3.5]
        indptr = torch.arange(0, (batch_size + 1) * seq_len, seq_len,
                              dtype=torch.int32, device=self.device)
        kwargs = {}
        if chunk_size is not None:  # [M3]
            kwargs["custom_mask"] = self._custom_mask(
                [seq_len] * batch_size, chunk_size)
            # [M3] split-kv scheduling depends on TOTAL batch workload, which
            # makes a doc's output vary with batch composition (measured
            # ~4e-3/layer, ~0.09 mel after 10 euler steps → ~1.3 audio after
            # hift). Streaming batching promises B=1 == B=N per doc (Task 3
            # gate B), so pin it off — 只作用于流式 plan；offline 路径保留
            # split-kv（其 parity 门本就是容差制，且小 chunk 下关 split-kv 的
            # SM 占用代价在 Task 5 基准中观测）。
            kwargs["disable_split_kv"] = True
        self.wrapper.plan(
            indptr, indptr, self.num_heads, self.num_heads, self.head_dim,
            causal=False, sm_scale=self.head_dim ** -0.5,
            q_data_type=dtype, kv_data_type=dtype, **kwargs,
        )
        self._planned_key = key

    def plan_docs(self, doc_lens: List[int], dtype: torch.dtype,
                  chunk_size: Optional[int] = None):  # [M3]
        """Plan for variable-length packed documents."""
        key = (tuple(doc_lens), dtype, chunk_size)  # [M3] key incl. chunk
        if key == self._planned_key:
            return
        self.plan_calls += 1  # [M3.5]
        indptr = torch.zeros(len(doc_lens) + 1, dtype=torch.int32, device=self.device)
        indptr[1:] = torch.cumsum(
            torch.tensor(doc_lens, dtype=torch.int32, device=self.device), dim=0)
        kwargs = {}
        if chunk_size is not None:  # [M3]
            kwargs["custom_mask"] = self._custom_mask(doc_lens, chunk_size)
            # [M3] see plan(): batch-composition-invariant per-doc outputs
            # (stream_step_batched parity gate) require split-kv off.
            kwargs["disable_split_kv"] = True
        self.wrapper.plan(
            indptr, indptr, self.num_heads, self.num_heads, self.head_dim,
            causal=False, sm_scale=self.head_dim ** -0.5,
            q_data_type=dtype, kv_data_type=dtype, **kwargs,
        )
        self._planned_key = key


def _rotate_half_interleaved(x):
    # x_transformers rotate_half: interleaved pairs (GPT-NeoX style)
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class FlashInferDiT(nn.Module):
    """State-dict compatible rewrite of cosyvoice.flow.DiT.dit.DiT (offline)."""

    def __init__(self, dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                 mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
                 enable_cuda_graph=False, cuda_graph_buckets=None,
                 device="cuda:0", static_chunk_size: int = 50,
                 plan_cache_size: int = 4,
                 stream_graph_buckets: Optional[List[int]] = None):
        super().__init__()
        # [M3] chunk size for the streaming chunk-causal mask (mel frames,
        # = vendored DiT.static_chunk_size = token chunk 25 * mel ratio 2)
        self._chunk_size = static_chunk_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.out_channels = out_channels
        self.enable_cuda_graph = enable_cuda_graph
        # mel frames (50 fps); buckets given in audio seconds
        self.cuda_graph_buckets = (
            sorted(int(d * 50) for d in cuda_graph_buckets) if cuda_graph_buckets else None
        )
        # [M3.5-r2] streaming bucketed CUDA graphs: buckets given directly in
        # mel FRAMES (50 fps; streaming shapes are chunk-quantized, not
        # duration-shaped). Only the b==2 single-session streaming CFG batch
        # hits this path (packed batch>1 keeps the flashinfer ragged route).
        # Dense-SDPA-in-graph is NOT bit-wise identical to the flashinfer
        # ragged eager path — opt-in knob; the M3 bit-exact interleave gates
        # keep running with this OFF. Quality arbiter: ASR CER gate.
        self.stream_graph_buckets = (
            sorted(int(n) for n in stream_graph_buckets)
            if stream_graph_buckets else None
        )

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=0.1)
             for _ in range(depth)]
        )
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        # [M3.5] pooled plan cache: the flashinfer wrapper holds ONE plan, so
        # reusing a plan across keys requires one WRAPPER (runner) per key.
        # Streaming chunk keys recur across requests (per-voice deterministic
        # shapes) but the old single runner replanned chunk-1 EVERY request —
        # later chunks of each request evicted its entry (~17ms/request:
        # ~14ms wrapper.plan host sync + ~2.4ms mask build). Bounded FIFO of
        # ready-planned runners, keyed by plan key; evicted runners are
        # RECYCLED (their 64MB workspace is reused, only replanned).
        # Memory: plan_cache_size=4 × 64MB workspace = 256MB (+ each runner's
        # single-slot mask, ~2*T^2 bool worst case). plan_cache_size=1
        # degenerates to the old single-runner replan-on-change behavior.
        self._plan_cache_size = max(1, plan_cache_size)
        self._runner_pool: OrderedDict = OrderedDict()  # plan key -> runner
        self._runner_device = torch.device(device)
        self._graph_cache = {}
        self._pack_cache = {}
        self._finalized = False
        # triton fusion only pays off when its launcher overhead is hidden by
        # graph replay; eager mode keeps the native-kernel path
        self._fused_tail = ((enable_cuda_graph or bool(stream_graph_buckets))
                            and _HAS_TRITON)
        self._ln_mod = (_ln_modulate_triton if self._fused_tail
                        else _ln_modulate_torch)

    # ------------------------------------------------------------------
    # [M3.5] pooled plan cache (see __init__ comment)
    def _pooled_runner(self, key):
        r = self._runner_pool.get(key)
        if r is None:
            if len(self._runner_pool) >= self._plan_cache_size:
                # FIFO evict + recycle: reuse the evicted runner's workspace;
                # its stale _planned_key won't match, so plan() below replans.
                _, r = self._runner_pool.popitem(last=False)
            else:
                r = RaggedAttentionRunner(self.heads, self.dim_head,
                                          self._runner_device)
            self._runner_pool[key] = r
        return r

    def _planned_runner(self, batch_size, seq_len, dtype, chunk_size=None):
        r = self._pooled_runner(("bs", batch_size, seq_len, dtype, chunk_size))
        r.plan(batch_size, seq_len, dtype, chunk_size=chunk_size)  # no-op on hit
        return r

    def _planned_runner_docs(self, doc_lens, dtype, chunk_size=None):
        r = self._pooled_runner(("docs", tuple(doc_lens), dtype, chunk_size))
        r.plan_docs(doc_lens, dtype, chunk_size=chunk_size)  # no-op on hit
        return r

    def finalize_weights(self):
        """Derive fused inference weights; call once after load + cast."""
        assert not self._finalized
        self._finalized = True
        for block in self.transformer_blocks:
            attn = block.attn
            attn._fi_w_qkv = torch.cat(
                [attn.to_q.weight, attn.to_k.weight, attn.to_v.weight], dim=0)
            attn._fi_b_qkv = torch.cat(
                [attn.to_q.bias, attn.to_k.bias, attn.to_v.bias], dim=0)
            # fold modulate's "+1" into the adaLN bias: chunk order is
            # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            bias = block.attn_norm.linear.bias.data.view(6, self.dim)
            bias[1] += 1.0
            bias[4] += 1.0
        # AdaLayerNormZero_Final chunk order is (scale, shift)
        self.norm_out.linear.bias.data.view(2, self.dim)[0] += 1.0

        # partial-RoPE tables (x_transformers semantics, fp32 math). The
        # interleaved-pair rotation is exactly a complex multiply:
        # (a+bi)(cos+isin) -> even' = a cos - b sin, odd' = b cos + a sin.
        with torch.autocast(device_type="cuda", enabled=False):
            freqs, _ = self.rotary_embed.forward_from_seq_len(_ROPE_MAX_LEN)
        freqs = freqs.reshape(-1, self.dim_head).float()
        self._rope_cos = freqs.cos()  # (L, 64), kept for the reference path
        self._rope_sin = freqs.sin()
        theta = freqs[:, 0::2]  # de-interleave: theta_i repeated pairwise
        self._rope_cis = torch.polar(torch.ones_like(theta), theta)  # (L, 32) c64
        self._rope_cos32 = theta.cos().contiguous()  # (L, 32) for the triton kernel
        self._rope_sin32 = theta.sin().contiguous()

        # one GEMM for every block's adaLN (+ the final one): t_emb is shared,
        # so 23 tiny per-block GEMMs collapse into a single fused projection
        # whose outputs are consumed as zero-cost slices.
        ada_w = [blk.attn_norm.linear.weight for blk in self.transformer_blocks]
        ada_b = [blk.attn_norm.linear.bias for blk in self.transformer_blocks]
        self._ada_w = torch.cat(ada_w + [self.norm_out.linear.weight], dim=0)
        self._ada_b = torch.cat(ada_b + [self.norm_out.linear.bias], dim=0)

        # conv position embedding as im2col + bmm: ~1.4x faster than cudnn's
        # grouped-conv kernel at these shapes and avoids its NCHW<->NHWC
        # layout conversions. weight (C, C/G, K) -> (G, C/G_out, K*C/G_in),
        # matching the unfolded (tap, cin) window layout.
        self._conv_pos = []
        cpe = self.input_embed.conv_pos_embed
        for seq in (cpe.conv1, cpe.conv2):
            conv = seq[0]
            g = conv.groups
            cg = conv.out_channels // g
            k = conv.kernel_size[0]
            w = conv.weight.view(g, cg, cg, k).permute(0, 1, 3, 2).reshape(g, cg, k * cg)
            self._conv_pos.append((w.transpose(1, 2).contiguous(), conv.bias, g, cg, k))

    # ------------------------------------------------------------------
    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        # [M3] streaming supported via chunk-causal custom mask (no assert)
        # run in pure fp16 regardless of the caller's autocast (like the TRT engine)
        with torch.autocast(device_type="cuda", enabled=False):
            dtype = self.proj_out.weight.dtype
            x, mu, spks, cond, t = (
                x.to(dtype), mu.to(dtype), spks.to(dtype), cond.to(dtype), t.to(dtype))

            b, _, seq_len = x.shape
            # [M3.5-r2] streaming bucketed graphs: b==2 == ONE session's CFG
            # pair (serial stream_step and packed B=1 both produce it), whose
            # mask is all-ones → true length == seq_len. seq > max bucket
            # falls through to the eager streaming path below.
            if (streaming and b == 2 and self.stream_graph_buckets is not None
                    and seq_len <= self.stream_graph_buckets[-1]):
                return self._forward_graph_bucketed_stream(x, mu, t, spks, cond)
            # [M3] CUDA graphs capture a fixed (mask-free) plan: offline only
            if self.enable_cuda_graph and b == 2 and not streaming:
                if self.cuda_graph_buckets is not None:
                    return self._forward_graph_bucketed(x, mu, t, spks, cond)
                return self._forward_graph(x, mu, t, spks, cond)
            if _HAS_TRITON:
                # packed varlen path: fastest no-graph route even at b=2
                # (single-sample CFG) — fused triton kernels amortize their
                # launcher cost over far fewer total launches
                return self._forward_packed(x, mask, mu, t, spks, cond,
                                            streaming=streaming)  # [M3]

            runner = self._planned_runner(
                b, seq_len, dtype,
                chunk_size=self._chunk_size if streaming else None)  # [M3]
            return self._forward_impl(x, mu, t, spks, cond, runner, None)

    def _forward_packed(self, x, mask, mu, t, spks, cond, streaming=False):
        """Batch>1 path: padded (2B, 80, maxT) rows are packed into one
        varlen sequence (total real tokens) for the transformer stack —
        zero padding compute, exact ragged attention per document."""
        assert _HAS_TRITON, "packed batch mode requires triton"
        b, _, maxT = x.shape
        n_dim = self.dim
        lens_t = mask[:, 0].sum(-1).to(torch.int64)
        lens = [int(v) for v in lens_t.tolist()]
        key = (b, maxT, tuple(lens))
        meta = self._pack_cache.get(key)
        if meta is None:
            device = x.device
            pack_idx = torch.cat([
                torch.arange(r * maxT, r * maxT + l, device=device)
                for r, l in enumerate(lens)])
            doc_ids = torch.repeat_interleave(
                torch.arange(b, device=device, dtype=torch.int32), lens_t.to(device))
            pos_ids = torch.cat([
                torch.arange(l, device=device, dtype=torch.int32) for l in lens])
            meta = {"pack_idx": pack_idx, "doc_ids": doc_ids, "pos_ids": pos_ids,
                    "lens": lens}
            # [M3] 有界化：流式 packed 模式下前缀逐 chunk 增长 × 批组成多样，
            # key 空间比 offline 大得多（~100KB/key 的 GPU 索引张量会无限累积）。
            # FIFO 淘汰足够——命中模式以"最近形状"为主，与 mask/plan 单槽同理。
            if len(self._pack_cache) >= 64:
                self._pack_cache.pop(next(iter(self._pack_cache)))
            self._pack_cache[key] = meta
        runner = self._planned_runner_docs(
            meta["lens"], x.dtype,
            chunk_size=self._chunk_size if streaming else None)  # [M3]

        # input embedding on the padded layout (cheap; causal conv is
        # pad-safe since padding sits at the tail of each row)
        xT, muT, condT = x.transpose(1, 2), mu.transpose(1, 2), cond.transpose(1, 2)
        spks_rep = spks.unsqueeze(1).expand(-1, maxT, -1)
        h0 = self.input_embed.proj(torch.cat([xT, condT, muT, spks_rep], dim=-1))
        h = self._conv_pos_forward(h0) + h0

        # pack: (2B, maxT, dim) -> (total, dim)
        hp = h.reshape(b * maxT, n_dim).index_select(0, meta["pack_idx"])
        total = hp.shape[0]
        doc_ids, pos_ids = meta["doc_ids"], meta["pos_ids"]

        t_emb = self.time_embed(t)  # (2B, dim)
        ada = F.linear(F.silu(t_emb), self._ada_w, self._ada_b)  # (2B, W)
        six = 6 * n_dim
        mods = [ada[:, i * six:(i + 1) * six].chunk(6, dim=-1)
                for i in range(self.depth)]
        fscale, fshift = ada[:, self.depth * six:].chunk(2, dim=-1)
        stride = ada.stride(0)
        eps = 1e-6

        def ln_mod(hh, shift, scale):
            out = torch.empty_like(hh)
            _ln_modulate_packed_kernel[(total,)](
                hh, shift, scale, out, doc_ids, stride, eps, n_dim)
            return out

        def gate_ln(hh, gate, y, shift, scale):
            h_out = torch.empty_like(hh)
            norm_out = torch.empty_like(hh)
            _gate_ln_modulate_packed_kernel[(total,)](
                hh, gate, y, shift, scale, h_out, norm_out, doc_ids, stride,
                eps, n_dim)
            return h_out, norm_out

        norm = ln_mod(hp, mods[0][0], mods[0][1])
        for i, block in enumerate(self.transformer_blocks):
            _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods[i]
            attn = block.attn
            qkv = F.linear(norm, attn._fi_w_qkv, attn._fi_b_qkv)  # (total, 3d)
            q = torch.empty(total, self.heads, self.dim_head,
                            dtype=qkv.dtype, device=qkv.device)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
            _qkv_rope_repack_packed_kernel[(total,)](
                qkv, q, k, v, self._rope_cos32, self._rope_sin32, pos_ids, n_dim)
            attn_out = attn.to_out[0](
                runner.wrapper.run(q, k, v).reshape(total, n_dim))
            hp, ffn_in = gate_ln(hp, gate_msa, attn_out, shift_mlp, scale_mlp)
            ff_out = block.ff(ffn_in)
            if i + 1 < self.depth:
                hp, norm = gate_ln(hp, gate_mlp, ff_out, mods[i + 1][0], mods[i + 1][1])
            else:
                _, norm = gate_ln(hp, gate_mlp, ff_out, fshift, fscale)

        y = self.proj_out(norm)  # (total, 80)
        out = torch.zeros(b * maxT, y.shape[-1], dtype=y.dtype, device=y.device)
        out.index_copy_(0, meta["pack_idx"], y)
        return out.view(b, maxT, -1).transpose(1, 2)

    def _apply_rope_head0(self, x_bnhd, cis):
        """x_transformers partial rope == rotate only head 0 in NHD layout.
        x_bnhd: (b, n, H, D); cis: (n, D/2) complex64, broadcast over batch."""
        b, n = x_bnhd.shape[:2]
        x0 = torch.view_as_complex(
            x_bnhd[:, :, 0].float().reshape(b, n, self.dim_head // 2, 2))
        x_bnhd[:, :, 0] = torch.view_as_real(x0 * cis).flatten(-2).to(x_bnhd.dtype)

    def _attention(self, attn, x, cis, runner, pad_mask=None):
        b, n, _ = x.shape
        qkv = F.linear(x, attn._fi_w_qkv, attn._fi_b_qkv)  # (b, n, 3*dim)
        if self._fused_tail:
            # one triton kernel: split qkv into packed q/k/v AND rotate head 0
            # (replaces 3 .contiguous() copies + the rope cast/mul chain)
            rows = b * n
            q = torch.empty(rows, self.heads, self.dim_head,
                            dtype=qkv.dtype, device=qkv.device)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
            _qkv_rope_repack_kernel[(rows,)](
                qkv.view(rows, 3 * self.dim), q, k, v,
                self._rope_cos32, self._rope_sin32, n, self.dim)
            if runner is not None:
                out = runner.wrapper.run(q, k, v)  # (b*n, H, D)
            else:
                out = F.scaled_dot_product_attention(
                    q.view(b, n, self.heads, self.dim_head).transpose(1, 2),
                    k.view(b, n, self.heads, self.dim_head).transpose(1, 2),
                    v.view(b, n, self.heads, self.dim_head).transpose(1, 2),
                    attn_mask=pad_mask)
                out = out.transpose(1, 2)
            return attn.to_out[0](out.reshape(b, n, self.dim))

        # rope folded into the packing copy we must pay anyway: rotate the
        # head-0 slice (complex multiply) and cat with the untouched tail —
        # cat emits the densely packed tensor directly, replacing the
        # index_put + .contiguous() chain (4 kernels/tensor instead of 6).
        q, k, v = qkv.chunk(3, dim=-1)  # (b, n, dim) strided views

        def rope_pack(x):
            x0 = torch.view_as_complex(x[..., :64].float().reshape(b, n, 32, 2))
            x0 = torch.view_as_real(x0 * cis).flatten(-2).to(x.dtype)
            return torch.cat([x0, x[..., 64:]], dim=-1)

        q = rope_pack(q)
        k = rope_pack(k)
        if runner is not None:
            out = runner.wrapper.run(
                q.view(b * n, self.heads, self.dim_head),
                k.view(b * n, self.heads, self.dim_head),
                v.reshape(b * n, self.heads, self.dim_head).contiguous(),
            )  # (b*n, H, D)
        else:
            # bucketed-graph path: both CFG docs share one real length, so a
            # single SDPA with a runtime-updated key-padding mask is exact
            out = F.scaled_dot_product_attention(
                q.view(b, n, self.heads, self.dim_head).transpose(1, 2),
                k.view(b, n, self.heads, self.dim_head).transpose(1, 2),
                v.reshape(b, n, self.heads, self.dim_head).transpose(1, 2),
                attn_mask=pad_mask)
            out = out.transpose(1, 2)
        return attn.to_out[0](out.reshape(b, n, self.dim))

    def _conv_pos_forward(self, h):
        """CausalConvPositionEmbedding via im2col + bmm on (b, n, c)."""
        b, n, c = h.shape
        y = h
        for w_t, bias, g, cg, k in self._conv_pos:
            xp = F.pad(y.transpose(1, 2), (k - 1, 0))     # (b, c, n+k-1)
            xu = (xp.view(b, g, cg, n + k - 1)
                  .unfold(3, k, 1)                        # (b, g, cg, n, k)
                  .permute(1, 0, 3, 4, 2)
                  .reshape(g, b * n, k * cg))
            y = torch.bmm(xu, w_t).view(g, b, n, cg).permute(1, 2, 0, 3).reshape(b, n, c)
            y = F.mish(y + bias)
        return y

    def _forward_impl(self, x, mu, t, spks, cond, runner, pad_mask):
        # x/mu/cond: (b, 80, n); t: (b,); spks: (b, 80)
        xT = x.transpose(1, 2)
        muT = mu.transpose(1, 2)
        condT = cond.transpose(1, 2)
        n = xT.shape[1]

        t_emb = self.time_embed(t)  # (b, dim)
        spks_rep = spks.unsqueeze(1).expand(-1, n, -1)
        h0 = self.input_embed.proj(torch.cat([xT, condT, muT, spks_rep], dim=-1))
        h = self._conv_pos_forward(h0) + h0

        cis = self._rope_cis[:n]
        # all 23 adaLN projections in one GEMM ("+1" already folded into bias)
        ada = F.linear(F.silu(t_emb), self._ada_w, self._ada_b).unsqueeze(1)

        six = 6 * self.dim
        mods = [ada[:, :, i * six:(i + 1) * six].chunk(6, dim=-1)
                for i in range(self.depth)]
        fscale, fshift = ada[:, :, self.depth * six:].chunk(2, dim=-1)

        if self._fused_tail:
            # gated residuals fused into the NEXT norm+modulate (2 stores/kernel)
            norm = self._ln_mod(h, mods[0][0], mods[0][1])
            for i, block in enumerate(self.transformer_blocks):
                _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods[i]
                attn_out = self._attention(block.attn, norm, cis, runner, pad_mask)
                h, ffn_in = _gate_ln_modulate(h, gate_msa, attn_out, shift_mlp, scale_mlp)
                ff_out = block.ff(ffn_in)
                if i + 1 < self.depth:
                    h, norm = _gate_ln_modulate(h, gate_mlp, ff_out,
                                                mods[i + 1][0], mods[i + 1][1])
                else:
                    _, norm = _gate_ln_modulate(h, gate_mlp, ff_out, fshift, fscale)
            return self.proj_out(norm).transpose(1, 2)

        for i, block in enumerate(self.transformer_blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods[i]
            norm = self._ln_mod(h, shift_msa, scale_msa)
            attn_out = self._attention(block.attn, norm, cis, runner, pad_mask)
            h = torch.addcmul(h, gate_msa, attn_out)
            ffn_in = self._ln_mod(h, shift_mlp, scale_mlp)
            h = torch.addcmul(h, gate_mlp, block.ff(ffn_in))

        h = self._ln_mod(h, fshift, fscale)
        return self.proj_out(h).transpose(1, 2)

    # ------------------------------------------------------------------
    def _forward_graph(self, x, mu, t, spks, cond):
        key = (x.shape[0], x.shape[2])
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture(x.shape[0], x.shape[2], x.dtype, bucket=False)
            self._graph_cache[key] = entry
        return self._replay(entry, x, mu, t, spks, cond, x.shape[2])

    def _forward_graph_bucketed(self, x, mu, t, spks, cond):
        b, _, n = x.shape
        bucket = next((s for s in self.cuda_graph_buckets if s >= n), None)
        if bucket is None:  # longer than the largest bucket: eager fallback
            runner = self._planned_runner(b, n, x.dtype)
            return self._forward_impl(x, mu, t, spks, cond, runner, None)
        key = ("bucket", b, bucket)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture(b, bucket, x.dtype, bucket=True)
            self._graph_cache[key] = entry
        entry["pad_mask"][..., :n] = True
        entry["pad_mask"][..., n:] = False
        return self._replay(entry, x, mu, t, spks, cond, n)

    def _forward_graph_bucketed_stream(self, x, mu, t, spks, cond):
        """[M3.5-r2] streaming bucket replay. For a padded bucket length N the
        chunk-causal grid `k//C <= q//C` is position-only (CONSTANT per
        bucket); the per-call variation is only the true-length key padding,
        so the runtime mask update is `grid & (k < n)` written into the
        static (1,1,N,N) bool buffer the captured SDPA reads (~N^2 bytes,
        ~1MB at N=1024 — trivial vs the ~50ms of launch overhead removed).
        Rows q >= n attend to a superset of keys but are sliced off by
        _replay; both CFG docs share one true length so one mask broadcasts
        over the batch dim."""
        b, _, n = x.shape
        bucket = next(s for s in self.stream_graph_buckets if s >= n)
        key = ("sbucket", b, bucket)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture(b, bucket, x.dtype, bucket=True, stream=True)
            self._graph_cache[key] = entry
        torch.logical_and(entry["grid"], entry["k_idx"] < n,
                          out=entry["pad_mask"][0, 0])
        return self._replay(entry, x, mu, t, spks, cond, n)

    def _replay(self, entry, x, mu, t, spks, cond, n):
        s = entry["x"]
        s["x"][:, :, :n].copy_(x)
        s["x"][:, :, n:].zero_()
        s["mu"][:, :, :n].copy_(mu)
        s["mu"][:, :, n:].zero_()
        s["cond"][:, :, :n].copy_(cond)
        s["cond"][:, :, n:].zero_()
        s["t"].copy_(t)
        s["spks"].copy_(spks)
        entry["graph"].replay()
        return entry["out"][:, :, :n]

    def _capture(self, b, n, dtype, bucket, stream=False):
        device = self.proj_out.weight.device
        # [M3.5] pin the CUDA device for the whole capture: Stream()/
        # current_stream()/CUDAGraph capture/synchronize() all target the
        # CURRENT device — with --token2wav-device cuda:1 (LLM on cuda:0)
        # the un-pinned version captures on the wrong device's stream.
        with torch.cuda.device(device):
            return self._capture_impl(b, n, dtype, bucket, stream, device)

    def _capture_impl(self, b, n, dtype, bucket, stream, device):
        static = {
            "x": torch.zeros(b, 80, n, dtype=dtype, device=device),
            "mu": torch.zeros(b, 80, n, dtype=dtype, device=device),
            "cond": torch.zeros(b, 80, n, dtype=dtype, device=device),
            "t": torch.zeros(b, dtype=dtype, device=device),
            "spks": torch.zeros(b, 80, dtype=dtype, device=device),
        }
        grid = k_idx = None
        if bucket and stream:
            # [M3.5-r2] streaming bucket: full 2D (1,1,N,N) mask buffer
            # (chunk-causal grid needs per-query rows, unlike the offline
            # key-padding (1,1,1,N)); grid/k_idx are kept for the runtime
            # `grid & (k < n)` update in _forward_graph_bucketed_stream.
            runner = None
            idx = torch.arange(n, device=device)
            grid = (idx.view(1, -1) // self._chunk_size
                    <= idx.view(-1, 1) // self._chunk_size)
            k_idx = idx.view(1, -1)
            pad_mask = grid.clone().view(1, 1, n, n)
        elif bucket:
            runner = None
            pad_mask = torch.ones(1, 1, 1, n, dtype=torch.bool, device=device)
        else:
            # each captured graph bakes its plan's launch metadata: private runner
            runner = RaggedAttentionRunner(self.heads, self.dim_head, device)
            runner.plan(b, n, dtype)
            pad_mask = None

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._forward_impl(static["x"], static["mu"], static["t"],
                                   static["spks"], static["cond"], runner, pad_mask)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._forward_impl(static["x"], static["mu"], static["t"],
                                     static["spks"], static["cond"], runner, pad_mask)
        return {"graph": graph, "out": out, "x": static, "runner": runner,
                "pad_mask": pad_mask, "grid": grid, "k_idx": k_idx}


@torch.inference_mode()
def _solve_euler_batched(decoder, z, mu, mask, spks, cond, n_timesteps=10,
                         streaming=False):  # [M3] thread streaming flag
    """Batched CFG euler solver: the repo's solve_euler hardcodes batch=1
    buffers, so multi-sample batches build the 2B-row CFG stack here."""
    B = mu.shape[0]
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=spks.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t, dt = t_span[0], t_span[1] - t_span[0]

    mask_in = mask.repeat(2, 1, 1).to(spks.dtype)
    mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0).to(spks.dtype)
    spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
    cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0).to(spks.dtype)
    t_in = torch.zeros(2 * B, device=mu.device, dtype=spks.dtype)

    x = z.to(spks.dtype)
    for step in range(1, len(t_span)):
        x_in = x.repeat(2, 1, 1)
        t_in.fill_(t)
        dphi_dt = decoder.forward_estimator(
            x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming)  # [M3]
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [B, B], dim=0)
        dphi_dt = ((1.0 + decoder.inference_cfg_rate) * dphi_dt
                   - decoder.inference_cfg_rate * cfg_dphi_dt)
        x = x + dt * dphi_dt
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x


@torch.inference_mode()
def flow_inference_batched(flow, token_list, prompt_feat_list, embedding):
    """Batched replica of CausalMaskedDiffWithDiT.inference (offline).

    Args:
        flow: the CausalMaskedDiffWithDiT module (fp16, flashinfer estimator).
        token_list: per-sample [prompt_tokens + generated_tokens] (list of list[int]).
        prompt_feat_list: per-sample prompt mel (1, L_i, 80).
        embedding: (B, 192) speaker embeddings.
    Returns:
        list of per-sample generated mel (1, 80, mel_len2_i), fp32.
    """
    device = embedding.device
    from faster_cosyvoice.token2wav.cosyvoice.utils.mask import make_pad_mask

    embedding = embedding.to(next(flow.parameters()).dtype)
    B = len(token_list)
    token_lens = torch.tensor([len(tk) for tk in token_list], device=device)
    max_tok = int(token_lens.max())
    token = torch.zeros(B, max_tok, dtype=torch.long, device=device)
    for i, tk in enumerate(token_list):
        token[i, :len(tk)] = torch.tensor(tk, device=device)

    embedding = F.normalize(embedding, dim=1)
    embedding = flow.spk_embed_affine_layer(embedding)

    mask = (~make_pad_mask(token_lens)).unsqueeze(-1).to(embedding)
    token = flow.input_embedding(torch.clamp(token, min=0)) * mask
    h = flow.pre_lookahead_layer(token)  # zero right-pad == batch pad: exact
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)

    mel_lens = token_lens * flow.token_mel_ratio
    max_mel = int(mel_lens.max())
    conds = torch.zeros(B, max_mel, flow.output_size, device=device, dtype=h.dtype)
    mel_len1 = []
    for i, pf in enumerate(prompt_feat_list):
        l1 = pf.shape[1]
        mel_len1.append(l1)
        conds[i, :l1] = pf[0].to(h.dtype)
    conds = conds.transpose(1, 2)

    mel_mask = (~make_pad_mask(mel_lens, max_len=max_mel)).to(h)

    z = torch.randn(B, flow.output_size, max_mel, device=device, dtype=h.dtype)
    feat = _solve_euler_batched(
        flow.decoder, z,
        mu=h.transpose(1, 2).contiguous(),
        mask=mel_mask.unsqueeze(1),
        spks=embedding,
        cond=conds,
        n_timesteps=10,
    )
    return [feat[i:i + 1, :, mel_len1[i]:int(mel_lens[i])].float() for i in range(B)]


@torch.inference_mode()
def flow_inference_batched_streaming(flow, token_list, prompt_feat_list,
                                     embedding, finalize_list):
    """[M3] Batched replica of CausalMaskedDiffWithDiT.inference (STREAMING),
    per-row mirror of vendored flow.py:364-409 with streaming=True.

    Diffs vs flow_inference_batched (offline) above:
    1. finalize semantics (flow.py:386-389): a doc with finalize=False routes
       its last `flow.pre_lookahead_len` tokens as PreLookaheadLayer *context*
       (conv right-lookahead) — those tokens do NOT enter the mu sequence, so
       its mel length is (n_tokens - pre_lookahead_len) * token_mel_ratio.
       finalize=True docs get full offline-style processing.
    2. embedding/lookahead run per-doc, not on a padded batch: the context
       kwarg differs per doc, so the offline "zero right-pad == batch pad"
       equivalence no longer applies; per-doc also keeps the op sequence
       bit-identical to the single-request stream_step path.
    3. F.normalize on the fp32 speaker embedding BEFORE the fp16 affine
       (matching stream_step's autocast semantics: normalize is not an
       autocast-fp16 op; the offline batched variant casts to fp16 first).
    4. noise is the decoder's fixed rand_noise prefix (flow_matching.py:222),
       not torch.randn: full-prefix recompute across chunks must be
       deterministic, and this keeps B=1 parity with stream_step. All docs
       slice the same noise from position 0, so one expand serves the batch.
    5. _solve_euler_batched(..., streaming=True) -> forward_estimator gets
       streaming=True -> plan_docs(chunk_size) per-doc chunk-causal masks.

    Args:
        flow: CausalMaskedDiffWithDiT (fp16, flashinfer estimator).
        token_list: per-doc [prompt_tokens + generated_prefix] (list of
            list[int]) — pre-concatenated exactly like the offline batched
            entry point; vendored inference() concatenates prompt_token
            before token prior to any processing (flow.py:381), so this is
            equivalent and keeps one signature for both batched paths.
        prompt_feat_list: per-doc prompt mel (1, L_i, 80).
        embedding: (B, 192) fp32 speaker embeddings.
        finalize_list: per-doc finalize flag (see diff 1).
    Returns:
        list of per-doc FULL-PREFIX generated mel (1, 80, mel_len2_i), fp32,
        prompt part excluded (flow.py:407 slice); caller slices the new tail
        by token_offset * token_mel_ratio.
    """
    device = embedding.device
    from faster_cosyvoice.token2wav.cosyvoice.utils.mask import make_pad_mask

    B = len(token_list)
    dtype = next(flow.parameters()).dtype
    lookahead = flow.pre_lookahead_len

    # xvec projection (flow.py:377-378; fp32 normalize per diff 3)
    embedding = F.normalize(embedding.float(), dim=1)
    embedding = flow.spk_embed_affine_layer(embedding.to(dtype))

    # per-doc embedding lookup + lookahead conv (diffs 1-2). B=1 masks are
    # all-ones so the vendored `* mask` (flow.py:383) is an exact no-op here.
    h_list, mel_len1, mel_lens = [], [], []
    for i, tk in enumerate(token_list):
        tok = torch.tensor([tk], dtype=torch.long, device=device)
        emb_tok = flow.input_embedding(torch.clamp(tok, min=0))
        if finalize_list[i]:
            h_i = flow.pre_lookahead_layer(emb_tok)
        else:
            h_i = flow.pre_lookahead_layer(emb_tok[:, :-lookahead],
                                           context=emb_tok[:, -lookahead:])
        h_i = h_i.repeat_interleave(flow.token_mel_ratio, dim=1)
        h_list.append(h_i)
        mel_len1.append(prompt_feat_list[i].shape[1])
        mel_lens.append(h_i.shape[1])

    max_mel = max(mel_lens)
    mu = torch.zeros(B, max_mel, h_list[0].shape[-1], device=device,
                     dtype=dtype)
    conds = torch.zeros(B, max_mel, flow.output_size, device=device,
                        dtype=dtype)
    for i in range(B):
        mu[i, :mel_lens[i]] = h_list[i][0]
        conds[i, :mel_len1[i]] = prompt_feat_list[i][0].to(dtype)
    conds = conds.transpose(1, 2)

    mel_lens_t = torch.tensor(mel_lens, device=device)
    mel_mask = (~make_pad_mask(mel_lens_t, max_len=max_mel)).to(mu)

    # fixed noise (diff 4): same prefix of the same buffer for every doc
    z = flow.decoder.rand_noise[:, :, :max_mel].to(device).to(dtype)
    z = z.expand(B, -1, -1)

    feat = _solve_euler_batched(
        flow.decoder, z,
        mu=mu.transpose(1, 2).contiguous(),
        mask=mel_mask.unsqueeze(1),
        spks=embedding,
        cond=conds,
        n_timesteps=10,
        streaming=True,  # [M3] diff 5: chunk-causal masks in the estimator
    )
    return [feat[i:i + 1, :, mel_len1[i]:mel_lens[i]].float()
            for i in range(B)]


@torch.inference_mode()
def token2wav_forward_batched(model, generated_speech_tokens_list,
                              prompt_audios_list, prompt_audios_sample_rate):
    """Batched replica of CosyVoice3_Token2Wav.forward (offline): batched
    flow with the packed flashinfer estimator, per-sample hift vocoder."""
    assert all(sr == 16000 for sr in prompt_audios_sample_rate)
    prompt_speech_tokens_list = model.prompt_audio_tokenization(prompt_audios_list)
    prompt_mels, prompt_mels_lens = model.get_prompt_mels(
        prompt_audios_list, prompt_audios_sample_rate)
    spk_emb = model.get_spk_emb(prompt_audios_list).to(model.device)

    token_list, prompt_feat_list = [], []
    for i in range(len(generated_speech_tokens_list)):
        tok_len = min(int(prompt_mels_lens[i].item() / 2),
                      len(prompt_speech_tokens_list[i]))
        prompt_tokens = prompt_speech_tokens_list[i][:tok_len]
        token_list.append(prompt_tokens + generated_speech_tokens_list[i])
        prompt_feat_list.append(prompt_mels[i:i + 1, :2 * tok_len].to(model.device))

    mels = flow_inference_batched(model.flow, token_list, prompt_feat_list, spk_emb)

    wavs = []
    for mel in mels:
        wav, _ = model.hift.inference(speech_feat=mel, finalize=True)
        wavs.append(wav)
    return wavs


def apply_flashinfer(model, enable_cuda_graph=False, cuda_graph_buckets=None,
                     stream_graph_buckets=None):
    """Patch a CosyVoice3_Token2Wav instance: fp16 flow + flashinfer estimator."""
    model.flow.half()
    model.fp16 = True  # forward_flow autocast context, matching the TRT path
    ref = model.flow.decoder.estimator
    device = next(ref.parameters()).device

    fi = FlashInferDiT(enable_cuda_graph=enable_cuda_graph,
                       cuda_graph_buckets=cuda_graph_buckets, device=str(device),
                       # [M3.5-r2] streaming buckets in mel frames (opt-in)
                       stream_graph_buckets=stream_graph_buckets,
                       # [M3] keep the streaming chunk in sync with the weights
                       static_chunk_size=getattr(ref, "static_chunk_size", 50))
    missing, unexpected = fi.load_state_dict(ref.state_dict(), strict=False)
    missing = [k for k in missing if "rope" not in k]
    assert not missing and not unexpected, f"state_dict mismatch: {missing} {unexpected}"
    fi = fi.to(device=device, dtype=torch.float16).eval()
    fi.finalize_weights()
    model.flow.decoder.estimator = fi
    return model


@torch.inference_mode()
def self_test(model_dir="./Fun-CosyVoice3-0.5B-2512"):
    from token2wav_cosyvoice3 import CosyVoice3_Token2Wav

    model = CosyVoice3_Token2Wav(model_dir, enable_trt=False)
    model.flow.half()
    ref = model.flow.decoder.estimator  # torch DiT, fp16
    device = next(ref.parameters()).device

    for graph_mode in (False, True):
        fi = FlashInferDiT(enable_cuda_graph=graph_mode, device=str(device))
        fi.load_state_dict(ref.state_dict(), strict=False)
        fi = fi.to(device=device, dtype=torch.float16).eval()
        fi.finalize_weights()

        torch.manual_seed(0)
        for n in (200, 517, 900):
            x = torch.randn(2, 80, n, device=device, dtype=torch.float16)
            mask = torch.ones(2, 1, n, device=device, dtype=torch.float16)
            mu = torch.randn(2, 80, n, device=device, dtype=torch.float16)
            t = torch.rand(1, device=device, dtype=torch.float16).expand(2).contiguous()
            spks = torch.randn(2, 80, device=device, dtype=torch.float16)
            cond = torch.randn(2, 80, n, device=device, dtype=torch.float16)

            out_ref = ref(x, mask, mu, t, spks, cond, streaming=False)
            out_fi = fi(x, mask, mu, t, spks, cond)
            diff = (out_ref - out_fi).abs().max().item()
            rel = diff / out_ref.abs().max().item()
            print(f"graph={graph_mode} n={n}: max_abs={diff:.5f} rel={rel:.5f}")
            # This DiT runs hidden activations at magnitude ~6000 (fp16 ulp = 4!),
            # so two numerically-equivalent fp16 implementations with different
            # summation orders legitimately diverge by a few percent (layer-wise
            # deltas are 1-17 ulp). ref16-vs-ref32 looks tighter only because the
            # kernel order is identical. End-to-end ASR is the real quality gate.
            assert rel < 0.15, "flashinfer estimator diverges beyond fp16-order noise"
    print("self-test passed")


if __name__ == "__main__":
    self_test()
