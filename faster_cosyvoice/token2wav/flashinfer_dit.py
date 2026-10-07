# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-accelerated CosyVoice3 DiT flow estimator.

``FlashInferDiT`` has the same state-dict layout as the original CosyVoice3
DiT, so it loads ``flow.pt`` without converting the checkpoint.  The class
contains the model-level execution paths; low-level Triton kernels live in
``flashinfer_kernels.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding

from faster_cosyvoice.token2wav.cosyvoice.flow.DiT.dit import InputEmbedding
from faster_cosyvoice.token2wav.cosyvoice.flow.DiT.modules import (
    AdaLayerNormZero_Final,
    DiTBlock,
    TimestepEmbedding,
)
from faster_cosyvoice.token2wav.flashinfer_attention import (
    RaggedAttentionRunner,
    build_chunk_causal_mask,
)
from faster_cosyvoice.token2wav.flashinfer_flow import (
    flow_inference_batched,
    flow_inference_batched_streaming,
    solve_euler_batched,
    token2wav_forward_batched,
)
from faster_cosyvoice.token2wav.flashinfer_kernels import (
    HAS_TRITON,
    gate_residual_layer_norm_modulate,
    gate_residual_layer_norm_modulate_packed,
    layer_norm_modulate,
    layer_norm_modulate_packed,
    layer_norm_modulate_torch,
    split_qkv_apply_rope,
    split_qkv_apply_rope_packed,
)

__all__ = [
    "FlashInferDiT",
    "RaggedAttentionRunner",
    "apply_flashinfer",
    "flow_inference_batched",
    "flow_inference_batched_streaming",
    "token2wav_forward_batched",
]

_ROPE_MAX_LEN = 4096
_MEL_FRAMES_PER_SECOND = 50
_PARTIAL_ROPE_DIM = 64
_MAX_PACK_LAYOUT_CACHE_SIZE = 64
_FLOW_CHANNELS = 80
_SPEAKER_CONDITION_DIM = 80

# Kept as a module-level compatibility hook for the existing tests and for
# deployments that explicitly disable the Triton packed path.
_HAS_TRITON = HAS_TRITON

# Backward-compatible alias for existing callers and tests.
_chunk_causal_flat_mask = build_chunk_causal_mask
_solve_euler_batched = solve_euler_batched


@dataclass(slots=True)
class _PackedLayout:
    """GPU indices that map a padded batch to FlashInfer's packed layout."""

    document_lengths: tuple[int, ...]
    gather_indices: torch.Tensor
    document_ids: torch.Tensor
    positions: torch.Tensor


@dataclass(slots=True)
class _GraphEntry:
    """Static buffers and metadata owned by one captured CUDA Graph."""

    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    inputs: dict[str, torch.Tensor]
    runner: RaggedAttentionRunner | None
    padding_mask: torch.Tensor | None
    chunk_grid: torch.Tensor | None
    key_positions: torch.Tensor | None


class FlashInferDiT(nn.Module):
    """State-dict-compatible inference rewrite of the CosyVoice3 DiT.

    The model selects one of three execution paths:

    * packed FlashInfer attention for eager and multi-session inference;
    * exact-shape CUDA Graph replay for single-session offline inference;
    * bucketed CUDA Graph replay for configured streaming shapes.

    Args:
      dim:
        Transformer hidden size.
      depth:
        Number of DiT blocks.
      heads:
        Number of attention heads.
      dim_head:
        Dimension of one attention head.
      enable_cuda_graph:
        Enable the single-session offline CUDA Graph path.
      cuda_graph_buckets:
        Offline graph buckets expressed in seconds of audio.
      stream_graph_buckets:
        Streaming graph buckets expressed in Mel frames.
      static_chunk_size:
        Chunk size, in Mel frames, used by streaming causal attention.
      plan_cache_size:
        Maximum number of ready FlashInfer plans retained by the model.
    """

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: int = 2,
        mel_dim: int = 80,
        mu_dim: int = 80,
        spk_dim: int = 80,
        out_channels: int = 80,
        enable_cuda_graph: bool = False,
        cuda_graph_buckets: Sequence[float] | None = None,
        device: torch.device | str = "cuda:0",
        static_chunk_size: int = 50,
        plan_cache_size: int = 4,
        stream_graph_buckets: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self._chunk_size = static_chunk_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.out_channels = out_channels
        self.enable_cuda_graph = enable_cuda_graph
        self.cuda_graph_buckets = (
            sorted(int(duration * _MEL_FRAMES_PER_SECOND) for duration in cuda_graph_buckets)
            if cuda_graph_buckets
            else None
        )
        self.stream_graph_buckets = (
            sorted(int(length) for length in stream_graph_buckets) if stream_graph_buckets else None
        )

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=0.1,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self._plan_cache_size = max(1, plan_cache_size)
        self._runner_pool: OrderedDict[tuple[Any, ...], RaggedAttentionRunner] = OrderedDict()
        self._runner_device = torch.device(device)
        self._graph_cache: dict[tuple[Any, ...], _GraphEntry] = {}
        self._pack_cache: OrderedDict[tuple[Any, ...], _PackedLayout] = OrderedDict()
        self._finalized = False
        self._fused_tail = (enable_cuda_graph or bool(stream_graph_buckets)) and _HAS_TRITON
        self._layer_norm_modulate = (
            layer_norm_modulate if self._fused_tail else layer_norm_modulate_torch
        )

    def _pooled_runner(self, key: tuple[Any, ...]) -> RaggedAttentionRunner:
        """Return a prepared-runner slot, recycling the oldest slot if full."""
        runner = self._runner_pool.get(key)
        if runner is None:
            if len(self._runner_pool) >= self._plan_cache_size:
                _, runner = self._runner_pool.popitem(last=False)
            else:
                runner = RaggedAttentionRunner(self.heads, self.dim_head, self._runner_device)
            self._runner_pool[key] = runner
        return runner

    def _planned_runner(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> RaggedAttentionRunner:
        sequence_length = seq_len
        key = ("equal", batch_size, sequence_length, dtype, chunk_size)
        runner = self._pooled_runner(key)
        runner.plan(batch_size, sequence_length, dtype, chunk_size=chunk_size)
        return runner

    def _planned_runner_docs(
        self,
        doc_lens: Sequence[int],
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> RaggedAttentionRunner:
        document_lengths = doc_lens
        key = ("ragged", tuple(document_lengths), dtype, chunk_size)
        runner = self._pooled_runner(key)
        runner.plan_docs(document_lengths, dtype, chunk_size=chunk_size)
        return runner

    def finalize_weights(self) -> None:
        """Build inference-only fused weights after loading and casting.

        The source module remains state-dict compatible until this method is
        called.  Fused QKV, AdaLN, RoPE, and positional-convolution tensors are
        then derived once and reused by every forward pass.
        """
        if self._finalized:
            raise RuntimeError("FlashInferDiT weights have already been finalized")
        self._finalized = True
        for block in self.transformer_blocks:
            attention = block.attn
            attention._fi_w_qkv = torch.cat(
                [
                    attention.to_q.weight,
                    attention.to_k.weight,
                    attention.to_v.weight,
                ],
                dim=0,
            )
            attention._fi_b_qkv = torch.cat(
                [attention.to_q.bias, attention.to_k.bias, attention.to_v.bias],
                dim=0,
            )

            # Fold AdaLN's ``1 + scale`` into the scale bias. Chunk order is
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
            bias = block.attn_norm.linear.bias.data.view(6, self.dim)
            bias[1] += 1.0
            bias[4] += 1.0

        # AdaLayerNormZero_Final uses (scale, shift).
        self.norm_out.linear.bias.data.view(2, self.dim)[0] += 1.0

        # partial-RoPE tables (x_transformers semantics, fp32 math). The
        # interleaved-pair rotation is exactly a complex multiply:
        # (a+bi)(cos+isin) -> even' = a cos - b sin, odd' = b cos + a sin.
        with torch.autocast(device_type="cuda", enabled=False):
            freqs, _ = self.rotary_embed.forward_from_seq_len(_ROPE_MAX_LEN)
        freqs = freqs.reshape(-1, self.dim_head).float()
        self._rope_cos = freqs.cos()
        self._rope_sin = freqs.sin()
        theta = freqs[:, 0::2]
        self._rope_cis = torch.polar(torch.ones_like(theta), theta)
        self._rope_cos32 = theta.cos().contiguous()
        self._rope_sin32 = theta.sin().contiguous()

        # The time embedding is shared by every block, so their AdaLN
        # projections (plus the final AdaLN) collapse into one GEMM. Each
        # block then consumes a view into that fused projection.
        ada_weights = [block.attn_norm.linear.weight for block in self.transformer_blocks]
        ada_biases = [block.attn_norm.linear.bias for block in self.transformer_blocks]
        self._ada_w = torch.cat(ada_weights + [self.norm_out.linear.weight], dim=0)
        self._ada_b = torch.cat(ada_biases + [self.norm_out.linear.bias], dim=0)

        # conv position embedding as im2col + bmm: ~1.4x faster than cudnn's
        # grouped-conv kernel at these shapes and avoids its NCHW<->NHWC
        # layout conversions. weight (C, C/G, K) -> (G, C/G_out, K*C/G_in),
        # matching the unfolded (tap, cin) window layout.
        self._conv_pos = []
        conv_position = self.input_embed.conv_pos_embed
        for sequence in (conv_position.conv1, conv_position.conv2):
            convolution = sequence[0]
            num_groups = convolution.groups
            channels_per_group = convolution.out_channels // num_groups
            kernel_size = convolution.kernel_size[0]
            weight = convolution.weight.view(
                num_groups,
                channels_per_group,
                channels_per_group,
                kernel_size,
            )
            weight = weight.permute(0, 1, 3, 2).reshape(
                num_groups,
                channels_per_group,
                kernel_size * channels_per_group,
            )
            self._conv_pos.append(
                (
                    weight.transpose(1, 2).contiguous(),
                    convolution.bias,
                    num_groups,
                    channels_per_group,
                    kernel_size,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> torch.Tensor:
        """Estimate the flow field for offline or streaming synthesis."""
        if spks is None or cond is None:
            raise ValueError("spks and cond are required")

        # Match the TensorRT path: the estimator itself always runs in the
        # weight dtype, independent of the caller's autocast context.
        with torch.autocast(device_type="cuda", enabled=False):
            dtype = self.proj_out.weight.dtype
            x = x.to(dtype)
            mu = mu.to(dtype)
            spks = spks.to(dtype)
            cond = cond.to(dtype)
            t = t.to(dtype)

            batch_size, _, sequence_length = x.shape
            if (
                streaming
                and batch_size == 2
                and self.stream_graph_buckets is not None
                and sequence_length <= self.stream_graph_buckets[-1]
            ):
                return self._forward_graph_bucketed_stream(x, mu, t, spks, cond)

            if self.enable_cuda_graph and batch_size == 2 and not streaming:
                if self.cuda_graph_buckets is not None:
                    return self._forward_graph_bucketed(x, mu, t, spks, cond)
                return self._forward_graph(x, mu, t, spks, cond)

            if _HAS_TRITON:
                return self._forward_packed(x, mask, mu, t, spks, cond, streaming=streaming)

            runner = self._planned_runner(
                batch_size,
                sequence_length,
                dtype,
                chunk_size=self._chunk_size if streaming else None,
            )
            return self._forward_impl(x, mu, t, spks, cond, runner, None)

    def _get_packed_layout(
        self,
        document_lengths_tensor: torch.Tensor,
        padded_length: int,
        device: torch.device,
    ) -> _PackedLayout:
        document_lengths = tuple(int(length) for length in document_lengths_tensor.tolist())
        cache_key = (len(document_lengths), padded_length, document_lengths)
        layout = self._pack_cache.get(cache_key)
        if layout is not None:
            return layout

        gather_indices = torch.cat(
            [
                torch.arange(
                    row * padded_length,
                    row * padded_length + length,
                    device=device,
                )
                for row, length in enumerate(document_lengths)
            ]
        )
        document_ids = torch.repeat_interleave(
            torch.arange(len(document_lengths), device=device, dtype=torch.int32),
            document_lengths_tensor.to(device),
        )
        positions = torch.cat(
            [torch.arange(length, device=device, dtype=torch.int32) for length in document_lengths]
        )
        layout = _PackedLayout(
            document_lengths=document_lengths,
            gather_indices=gather_indices,
            document_ids=document_ids,
            positions=positions,
        )

        # Streaming prefixes create many shape combinations. A bounded FIFO
        # retains recurring recent layouts without accumulating GPU indices.
        if len(self._pack_cache) >= _MAX_PACK_LAYOUT_CACHE_SIZE:
            self._pack_cache.popitem(last=False)
        self._pack_cache[cache_key] = layout
        return layout

    def _embed_inputs(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Combine Flow inputs and apply the causal position convolution."""
        sequence_length = x.shape[2]
        speaker_condition = spks.unsqueeze(1).expand(-1, sequence_length, -1)
        projected = self.input_embed.proj(
            torch.cat(
                [
                    x.transpose(1, 2),
                    cond.transpose(1, 2),
                    mu.transpose(1, 2),
                    speaker_condition,
                ],
                dim=-1,
            )
        )
        return self._conv_pos_forward(projected) + projected

    def _project_modulations(
        self, t: torch.Tensor, *, add_sequence_axis: bool
    ) -> tuple[list[tuple[torch.Tensor, ...]], torch.Tensor, torch.Tensor]:
        """Project every block's AdaLN parameters with one matrix multiply."""
        time_embedding = self.time_embed(t)
        projected = F.linear(F.silu(time_embedding), self._ada_w, self._ada_b)
        if add_sequence_axis:
            projected = projected.unsqueeze(1)

        block_width = 6 * self.dim
        block_modulations = [
            projected[..., index * block_width : (index + 1) * block_width].chunk(6, dim=-1)
            for index in range(self.depth)
        ]
        final_scale, final_shift = projected[..., self.depth * block_width :].chunk(2, dim=-1)
        return block_modulations, final_scale, final_shift

    def _forward_packed(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        """Run a padded batch as packed variable-length documents."""
        assert _HAS_TRITON, "packed batch mode requires triton"
        batch_size, _, padded_length = x.shape
        document_lengths_tensor = mask[:, 0].sum(-1).to(torch.int64)
        layout = self._get_packed_layout(document_lengths_tensor, padded_length, x.device)
        runner = self._planned_runner_docs(
            layout.document_lengths,
            x.dtype,
            chunk_size=self._chunk_size if streaming else None,
        )

        hidden = self._embed_inputs(x, mu, spks, cond)
        packed_hidden = hidden.reshape(batch_size * padded_length, self.dim).index_select(
            0, layout.gather_indices
        )

        modulations, final_scale, final_shift = self._project_modulations(
            t, add_sequence_axis=False
        )
        modulation_stride = 6 * self.dim * self.depth + 2 * self.dim

        norm = layer_norm_modulate_packed(
            packed_hidden,
            modulations[0][0],
            modulations[0][1],
            layout.document_ids,
            modulation_stride,
        )
        for index, block in enumerate(self.transformer_blocks):
            _, _, attention_gate, feed_forward_shift, feed_forward_scale, feed_forward_gate = (
                modulations[index]
            )
            attention = block.attn
            qkv = F.linear(norm, attention._fi_w_qkv, attention._fi_b_qkv)
            query, key, value = split_qkv_apply_rope_packed(
                qkv,
                self._rope_cos32,
                self._rope_sin32,
                layout.positions,
                num_heads=self.heads,
                head_dim=self.dim_head,
                rope_dim=_PARTIAL_ROPE_DIM,
            )
            attention_output = attention.to_out[0](
                runner.wrapper.run(query, key, value).reshape(-1, self.dim)
            )
            packed_hidden, feed_forward_input = gate_residual_layer_norm_modulate_packed(
                packed_hidden,
                attention_gate,
                attention_output,
                feed_forward_shift,
                feed_forward_scale,
                layout.document_ids,
                modulation_stride,
            )
            feed_forward_output = block.ff(feed_forward_input)
            if index + 1 < self.depth:
                next_shift, next_scale = modulations[index + 1][:2]
                packed_hidden, norm = gate_residual_layer_norm_modulate_packed(
                    packed_hidden,
                    feed_forward_gate,
                    feed_forward_output,
                    next_shift,
                    next_scale,
                    layout.document_ids,
                    modulation_stride,
                )
            else:
                _, norm = gate_residual_layer_norm_modulate_packed(
                    packed_hidden,
                    feed_forward_gate,
                    feed_forward_output,
                    final_shift,
                    final_scale,
                    layout.document_ids,
                    modulation_stride,
                )

        packed_output = self.proj_out(norm)
        output = torch.zeros(
            batch_size * padded_length,
            packed_output.shape[-1],
            dtype=packed_output.dtype,
            device=packed_output.device,
        )
        output.index_copy_(0, layout.gather_indices, packed_output)
        return output.view(batch_size, padded_length, -1).transpose(1, 2)

    def _attention(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        rope: torch.Tensor,
        runner: RaggedAttentionRunner | None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run partial-RoPE attention through FlashInfer or graph-safe SDPA."""
        batch_size, sequence_length, _ = hidden.shape
        qkv = F.linear(hidden, attention._fi_w_qkv, attention._fi_b_qkv)

        if self._fused_tail:
            query, key, value = split_qkv_apply_rope(
                qkv,
                self._rope_cos32,
                self._rope_sin32,
                sequence_length=sequence_length,
                num_heads=self.heads,
                head_dim=self.dim_head,
                rope_dim=_PARTIAL_ROPE_DIM,
            )
            if runner is not None:
                output = runner.wrapper.run(query, key, value)
            else:
                output = F.scaled_dot_product_attention(
                    query.view(batch_size, sequence_length, self.heads, self.dim_head).transpose(
                        1, 2
                    ),
                    key.view(batch_size, sequence_length, self.heads, self.dim_head).transpose(
                        1, 2
                    ),
                    value.view(batch_size, sequence_length, self.heads, self.dim_head).transpose(
                        1, 2
                    ),
                    attn_mask=padding_mask,
                ).transpose(1, 2)
            return attention.to_out[0](output.reshape(batch_size, sequence_length, self.dim))

        query, key, value = qkv.chunk(3, dim=-1)

        def apply_partial_rope(tensor: torch.Tensor) -> torch.Tensor:
            rope_pairs = _PARTIAL_ROPE_DIM // 2
            rotated = torch.view_as_complex(
                tensor[..., :_PARTIAL_ROPE_DIM]
                .float()
                .reshape(batch_size, sequence_length, rope_pairs, 2)
            )
            rotated = torch.view_as_real(rotated * rope).flatten(-2).to(tensor.dtype)
            return torch.cat([rotated, tensor[..., _PARTIAL_ROPE_DIM:]], dim=-1)

        query = apply_partial_rope(query)
        key = apply_partial_rope(key)
        if runner is not None:
            output = runner.wrapper.run(
                query.view(batch_size * sequence_length, self.heads, self.dim_head),
                key.view(batch_size * sequence_length, self.heads, self.dim_head),
                value.reshape(batch_size * sequence_length, self.heads, self.dim_head).contiguous(),
            )
        else:
            output = F.scaled_dot_product_attention(
                query.view(batch_size, sequence_length, self.heads, self.dim_head).transpose(1, 2),
                key.view(batch_size, sequence_length, self.heads, self.dim_head).transpose(1, 2),
                value.reshape(batch_size, sequence_length, self.heads, self.dim_head).transpose(
                    1, 2
                ),
                attn_mask=padding_mask,
            ).transpose(1, 2)
        return attention.to_out[0](output.reshape(batch_size, sequence_length, self.dim))

    def _conv_pos_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate the causal position convolution as im2col plus BMM."""
        batch_size, sequence_length, num_channels = hidden.shape
        output = hidden
        for (
            transposed_weight,
            bias,
            num_groups,
            channels_per_group,
            kernel_size,
        ) in self._conv_pos:
            padded = F.pad(output.transpose(1, 2), (kernel_size - 1, 0))
            windows = (
                padded.view(
                    batch_size,
                    num_groups,
                    channels_per_group,
                    sequence_length + kernel_size - 1,
                )
                .unfold(3, kernel_size, 1)
                .permute(1, 0, 3, 4, 2)
                .reshape(
                    num_groups,
                    batch_size * sequence_length,
                    kernel_size * channels_per_group,
                )
            )
            output = torch.bmm(windows, transposed_weight)
            output = (
                output.view(
                    num_groups,
                    batch_size,
                    sequence_length,
                    channels_per_group,
                )
                .permute(1, 2, 0, 3)
                .reshape(batch_size, sequence_length, num_channels)
            )
            output = F.mish(output + bias)
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        runner: RaggedAttentionRunner | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = self._embed_inputs(x, mu, spks, cond)
        sequence_length = hidden.shape[1]
        rope = self._rope_cis[:sequence_length]
        modulations, final_scale, final_shift = self._project_modulations(t, add_sequence_axis=True)

        if self._fused_tail:
            norm = self._layer_norm_modulate(hidden, modulations[0][0], modulations[0][1])
            for index, block in enumerate(self.transformer_blocks):
                _, _, attention_gate, feed_forward_shift, feed_forward_scale, feed_forward_gate = (
                    modulations[index]
                )
                attention_output = self._attention(block.attn, norm, rope, runner, padding_mask)
                hidden, feed_forward_input = gate_residual_layer_norm_modulate(
                    hidden,
                    attention_gate,
                    attention_output,
                    feed_forward_shift,
                    feed_forward_scale,
                )
                feed_forward_output = block.ff(feed_forward_input)
                if index + 1 < self.depth:
                    next_shift, next_scale = modulations[index + 1][:2]
                    hidden, norm = gate_residual_layer_norm_modulate(
                        hidden,
                        feed_forward_gate,
                        feed_forward_output,
                        next_shift,
                        next_scale,
                    )
                else:
                    _, norm = gate_residual_layer_norm_modulate(
                        hidden,
                        feed_forward_gate,
                        feed_forward_output,
                        final_shift,
                        final_scale,
                    )
            return self.proj_out(norm).transpose(1, 2)

        for index, block in enumerate(self.transformer_blocks):
            (
                attention_shift,
                attention_scale,
                attention_gate,
                feed_forward_shift,
                feed_forward_scale,
                feed_forward_gate,
            ) = modulations[index]
            norm = self._layer_norm_modulate(hidden, attention_shift, attention_scale)
            attention_output = self._attention(block.attn, norm, rope, runner, padding_mask)
            hidden = torch.addcmul(hidden, attention_gate, attention_output)
            feed_forward_input = self._layer_norm_modulate(
                hidden, feed_forward_shift, feed_forward_scale
            )
            hidden = torch.addcmul(hidden, feed_forward_gate, block.ff(feed_forward_input))

        hidden = self._layer_norm_modulate(hidden, final_shift, final_scale)
        return self.proj_out(hidden).transpose(1, 2)

    def _forward_graph(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # Cache-key shapes are kept compatible with the original implementation
        # because profiling and regression tools inspect them directly.
        cache_key = (x.shape[0], x.shape[2])
        entry = self._graph_cache.get(cache_key)
        if entry is None:
            entry = self._capture(x.shape[0], x.shape[2], x.dtype, bucketed=False)
            self._graph_cache[cache_key] = entry
        return self._replay(entry, x, mu, t, spks, cond, x.shape[2])

    def _forward_graph_bucketed(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, sequence_length = x.shape
        assert self.cuda_graph_buckets is not None
        bucket = next(
            (candidate for candidate in self.cuda_graph_buckets if candidate >= sequence_length),
            None,
        )
        if bucket is None:
            runner = self._planned_runner(batch_size, sequence_length, x.dtype)
            return self._forward_impl(x, mu, t, spks, cond, runner, None)

        cache_key = ("bucket", batch_size, bucket)
        entry = self._graph_cache.get(cache_key)
        if entry is None:
            entry = self._capture(batch_size, bucket, x.dtype, bucketed=True)
            self._graph_cache[cache_key] = entry
        assert entry.padding_mask is not None
        entry.padding_mask[..., :sequence_length] = True
        entry.padding_mask[..., sequence_length:] = False
        return self._replay(entry, x, mu, t, spks, cond, sequence_length)

    def _forward_graph_bucketed_stream(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Replay a streaming graph with a runtime key-padding boundary."""
        batch_size, _, sequence_length = x.shape
        assert self.stream_graph_buckets is not None
        bucket = next(
            candidate for candidate in self.stream_graph_buckets if candidate >= sequence_length
        )
        cache_key = ("sbucket", batch_size, bucket)
        entry = self._graph_cache.get(cache_key)
        if entry is None:
            entry = self._capture(
                batch_size,
                bucket,
                x.dtype,
                bucketed=True,
                streaming=True,
            )
            self._graph_cache[cache_key] = entry

        assert entry.chunk_grid is not None
        assert entry.key_positions is not None
        assert entry.padding_mask is not None
        torch.logical_and(
            entry.chunk_grid,
            entry.key_positions < sequence_length,
            out=entry.padding_mask[0, 0],
        )
        return self._replay(entry, x, mu, t, spks, cond, sequence_length)

    def _replay(
        self,
        entry: _GraphEntry,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        static = entry.inputs
        static["x"][:, :, :sequence_length].copy_(x)
        static["x"][:, :, sequence_length:].zero_()
        static["mu"][:, :, :sequence_length].copy_(mu)
        static["mu"][:, :, sequence_length:].zero_()
        static["cond"][:, :, :sequence_length].copy_(cond)
        static["cond"][:, :, sequence_length:].zero_()
        static["t"].copy_(t)
        static["spks"].copy_(spks)
        entry.graph.replay()
        return entry.output[:, :, :sequence_length]

    def _capture(
        self,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        bucketed: bool,
        streaming: bool = False,
    ) -> _GraphEntry:
        device = self.proj_out.weight.device
        # CUDA stream creation and graph capture use the current device. Pin it
        # explicitly because Token2Wav can run on a device other than cuda:0.
        with torch.cuda.device(device):
            return self._capture_impl(
                batch_size,
                sequence_length,
                dtype,
                bucketed,
                streaming,
                device,
            )

    def _capture_impl(
        self,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        bucketed: bool,
        streaming: bool,
        device: torch.device,
    ) -> _GraphEntry:
        static = {
            "x": torch.zeros(
                batch_size,
                _FLOW_CHANNELS,
                sequence_length,
                dtype=dtype,
                device=device,
            ),
            "mu": torch.zeros(
                batch_size,
                _FLOW_CHANNELS,
                sequence_length,
                dtype=dtype,
                device=device,
            ),
            "cond": torch.zeros(
                batch_size,
                _FLOW_CHANNELS,
                sequence_length,
                dtype=dtype,
                device=device,
            ),
            "t": torch.zeros(batch_size, dtype=dtype, device=device),
            "spks": torch.zeros(
                batch_size,
                _SPEAKER_CONDITION_DIM,
                dtype=dtype,
                device=device,
            ),
        }
        chunk_grid = None
        key_positions = None
        if bucketed and streaming:
            runner = None
            positions = torch.arange(sequence_length, device=device)
            chunk_grid = (
                positions[None, :] // self._chunk_size <= positions[:, None] // self._chunk_size
            )
            key_positions = positions[None, :]
            padding_mask = chunk_grid.clone().view(1, 1, sequence_length, sequence_length)
        elif bucketed:
            runner = None
            padding_mask = torch.ones(
                1,
                1,
                1,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        else:
            runner = RaggedAttentionRunner(self.heads, self.dim_head, device)
            runner.plan(batch_size, sequence_length, dtype)
            padding_mask = None

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                self._forward_impl(
                    static["x"],
                    static["mu"],
                    static["t"],
                    static["spks"],
                    static["cond"],
                    runner,
                    padding_mask,
                )
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward_impl(
                static["x"],
                static["mu"],
                static["t"],
                static["spks"],
                static["cond"],
                runner,
                padding_mask,
            )
        return _GraphEntry(
            graph=graph,
            output=output,
            inputs=static,
            runner=runner,
            padding_mask=padding_mask,
            chunk_grid=chunk_grid,
            key_positions=key_positions,
        )


def apply_flashinfer(
    model: Any,
    enable_cuda_graph: bool = False,
    cuda_graph_buckets: Sequence[float] | None = None,
    stream_graph_buckets: Sequence[int] | None = None,
) -> Any:
    """Replace a Token2Wav model's Torch DiT with ``FlashInferDiT``.

    The Flow module is converted to FP16 first. The original estimator's state
    dict is then loaded without checkpoint conversion, after which the derived
    inference weights are finalized.
    """
    model.flow.half()
    model.fp16 = True
    reference_estimator = model.flow.decoder.estimator
    device = next(reference_estimator.parameters()).device

    estimator = FlashInferDiT(
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_buckets=cuda_graph_buckets,
        device=device,
        stream_graph_buckets=stream_graph_buckets,
        static_chunk_size=getattr(reference_estimator, "static_chunk_size", 50),
    )
    missing_keys, unexpected_keys = estimator.load_state_dict(
        reference_estimator.state_dict(), strict=False
    )
    missing_keys = [key for key in missing_keys if "rope" not in key]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "FlashInferDiT state-dict mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    estimator = estimator.to(device=device, dtype=torch.float16).eval()
    estimator.finalize_weights()
    model.flow.decoder.estimator = estimator
    return model


@torch.inference_mode()
def self_test(model_dir: str = "./Fun-CosyVoice3-0.5B-2512") -> None:
    """Compare the Torch and FlashInfer estimators on representative shapes."""
    from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

    model = CosyVoice3Token2Wav(model_dir, estimator_mode="torch")
    model.flow.half()
    reference_estimator = model.flow.decoder.estimator
    device = next(reference_estimator.parameters()).device

    for graph_mode in (False, True):
        estimator = FlashInferDiT(enable_cuda_graph=graph_mode, device=device)
        estimator.load_state_dict(reference_estimator.state_dict(), strict=False)
        estimator = estimator.to(device=device, dtype=torch.float16).eval()
        estimator.finalize_weights()

        torch.manual_seed(0)
        for sequence_length in (200, 517, 900):
            x = torch.randn(2, 80, sequence_length, device=device, dtype=torch.float16)
            mask = torch.ones(2, 1, sequence_length, device=device, dtype=torch.float16)
            mu = torch.randn(2, 80, sequence_length, device=device, dtype=torch.float16)
            time = torch.rand(1, device=device, dtype=torch.float16).expand(2).contiguous()
            spks = torch.randn(2, 80, device=device, dtype=torch.float16)
            cond = torch.randn(2, 80, sequence_length, device=device, dtype=torch.float16)

            reference_output = reference_estimator(x, mask, mu, time, spks, cond, streaming=False)
            output = estimator(x, mask, mu, time, spks, cond)
            max_difference = (reference_output - output).abs().max().item()
            relative_difference = max_difference / reference_output.abs().max().item()
            print(
                f"graph={graph_mode} frames={sequence_length}: "
                f"max_abs={max_difference:.5f} rel={relative_difference:.5f}"
            )
            # Different FP16 reduction orders can differ by several percent.
            # End-to-end ASR remains the final quality gate.
            assert relative_difference < 0.15
    print("self-test passed")


if __name__ == "__main__":
    self_test()
