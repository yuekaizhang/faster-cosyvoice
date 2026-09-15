# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels used by the FlashInfer DiT implementation.

The public helpers in this module describe tensor operations rather than kernel
launch details.  Keeping those details here makes :mod:`flashinfer_dit` read like
the model it implements.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _layer_norm_modulate_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        output_ptr,
        rows_per_batch,
        modulation_stride,
        eps: tl.constexpr,
        hidden_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, hidden_dim)

        hidden = tl.load(x_ptr + row * hidden_dim + columns).to(tl.float32)
        centered = hidden - tl.sum(hidden) / hidden_dim
        inverse_std = 1.0 / tl.sqrt(tl.sum(centered * centered) / hidden_dim + eps)

        batch_index = row // rows_per_batch
        shift = tl.load(shift_ptr + batch_index * modulation_stride + columns).to(tl.float32)
        scale = tl.load(scale_ptr + batch_index * modulation_stride + columns).to(tl.float32)
        output = centered * inverse_std * scale + shift
        tl.store(
            output_ptr + row * hidden_dim + columns,
            output.to(output_ptr.dtype.element_ty),
        )

    @triton.jit
    def _gate_residual_layer_norm_modulate_kernel(
        hidden_ptr,
        gate_ptr,
        residual_ptr,
        shift_ptr,
        scale_ptr,
        hidden_output_ptr,
        norm_output_ptr,
        rows_per_batch,
        modulation_stride,
        eps: tl.constexpr,
        hidden_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, hidden_dim)
        batch_index = row // rows_per_batch

        hidden = tl.load(hidden_ptr + row * hidden_dim + columns).to(tl.float32)
        gate = tl.load(gate_ptr + batch_index * modulation_stride + columns).to(tl.float32)
        residual = tl.load(residual_ptr + row * hidden_dim + columns).to(tl.float32)
        hidden = hidden + gate * residual
        tl.store(
            hidden_output_ptr + row * hidden_dim + columns,
            hidden.to(hidden_output_ptr.dtype.element_ty),
        )

        centered = hidden - tl.sum(hidden) / hidden_dim
        inverse_std = 1.0 / tl.sqrt(tl.sum(centered * centered) / hidden_dim + eps)
        shift = tl.load(shift_ptr + batch_index * modulation_stride + columns).to(tl.float32)
        scale = tl.load(scale_ptr + batch_index * modulation_stride + columns).to(tl.float32)
        output = centered * inverse_std * scale + shift
        tl.store(
            norm_output_ptr + row * hidden_dim + columns,
            output.to(norm_output_ptr.dtype.element_ty),
        )

    @triton.jit
    def _layer_norm_modulate_packed_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        output_ptr,
        document_ids_ptr,
        modulation_stride,
        eps: tl.constexpr,
        hidden_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, hidden_dim)

        hidden = tl.load(x_ptr + row * hidden_dim + columns).to(tl.float32)
        centered = hidden - tl.sum(hidden) / hidden_dim
        inverse_std = 1.0 / tl.sqrt(tl.sum(centered * centered) / hidden_dim + eps)

        document_index = tl.load(document_ids_ptr + row)
        shift = tl.load(shift_ptr + document_index * modulation_stride + columns).to(tl.float32)
        scale = tl.load(scale_ptr + document_index * modulation_stride + columns).to(tl.float32)
        output = centered * inverse_std * scale + shift
        tl.store(
            output_ptr + row * hidden_dim + columns,
            output.to(output_ptr.dtype.element_ty),
        )

    @triton.jit
    def _gate_residual_layer_norm_modulate_packed_kernel(
        hidden_ptr,
        gate_ptr,
        residual_ptr,
        shift_ptr,
        scale_ptr,
        hidden_output_ptr,
        norm_output_ptr,
        document_ids_ptr,
        modulation_stride,
        eps: tl.constexpr,
        hidden_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, hidden_dim)
        document_index = tl.load(document_ids_ptr + row)

        hidden = tl.load(hidden_ptr + row * hidden_dim + columns).to(tl.float32)
        gate = tl.load(gate_ptr + document_index * modulation_stride + columns).to(tl.float32)
        residual = tl.load(residual_ptr + row * hidden_dim + columns).to(tl.float32)
        hidden = hidden + gate * residual
        tl.store(
            hidden_output_ptr + row * hidden_dim + columns,
            hidden.to(hidden_output_ptr.dtype.element_ty),
        )

        centered = hidden - tl.sum(hidden) / hidden_dim
        inverse_std = 1.0 / tl.sqrt(tl.sum(centered * centered) / hidden_dim + eps)
        shift = tl.load(shift_ptr + document_index * modulation_stride + columns).to(tl.float32)
        scale = tl.load(scale_ptr + document_index * modulation_stride + columns).to(tl.float32)
        output = centered * inverse_std * scale + shift
        tl.store(
            norm_output_ptr + row * hidden_dim + columns,
            output.to(norm_output_ptr.dtype.element_ty),
        )

    @triton.jit
    def _split_qkv_apply_rope_kernel(
        qkv_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        rows_per_batch,
        hidden_dim: tl.constexpr,
        rope_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, hidden_dim)
        qkv_offset = row * 3 * hidden_dim

        query = tl.load(qkv_ptr + qkv_offset + columns).to(tl.float32)
        key = tl.load(qkv_ptr + qkv_offset + hidden_dim + columns).to(tl.float32)
        value = tl.load(qkv_ptr + qkv_offset + 2 * hidden_dim + columns)

        apply_rope = columns < rope_dim
        position = row % rows_per_batch
        pair = columns // 2
        cos = tl.load(
            rope_cos_ptr + position * (rope_dim // 2) + pair,
            mask=apply_rope,
            other=1.0,
        )
        sin = tl.load(
            rope_sin_ptr + position * (rope_dim // 2) + pair,
            mask=apply_rope,
            other=0.0,
        )
        partner = tl.where(columns % 2 == 0, columns + 1, columns - 1)
        sign = tl.where(columns % 2 == 0, -1.0, 1.0)
        query_partner = tl.load(qkv_ptr + qkv_offset + partner, mask=apply_rope, other=0.0).to(
            tl.float32
        )
        key_partner = tl.load(
            qkv_ptr + qkv_offset + hidden_dim + partner,
            mask=apply_rope,
            other=0.0,
        ).to(tl.float32)

        query = tl.where(apply_rope, query * cos + sign * query_partner * sin, query)
        key = tl.where(apply_rope, key * cos + sign * key_partner * sin, key)

        tl.store(
            query_ptr + row * hidden_dim + columns,
            query.to(query_ptr.dtype.element_ty),
        )
        tl.store(
            key_ptr + row * hidden_dim + columns,
            key.to(key_ptr.dtype.element_ty),
        )
        tl.store(value_ptr + row * hidden_dim + columns, value)

    @triton.jit
    def _split_qkv_apply_rope_packed_kernel(
        qkv_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        positions_ptr,
        hidden_dim: tl.constexpr,
        rope_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, hidden_dim)
        qkv_offset = row * 3 * hidden_dim

        query = tl.load(qkv_ptr + qkv_offset + columns).to(tl.float32)
        key = tl.load(qkv_ptr + qkv_offset + hidden_dim + columns).to(tl.float32)
        value = tl.load(qkv_ptr + qkv_offset + 2 * hidden_dim + columns)

        apply_rope = columns < rope_dim
        position = tl.load(positions_ptr + row)
        pair = columns // 2
        cos = tl.load(
            rope_cos_ptr + position * (rope_dim // 2) + pair,
            mask=apply_rope,
            other=1.0,
        )
        sin = tl.load(
            rope_sin_ptr + position * (rope_dim // 2) + pair,
            mask=apply_rope,
            other=0.0,
        )
        partner = tl.where(columns % 2 == 0, columns + 1, columns - 1)
        sign = tl.where(columns % 2 == 0, -1.0, 1.0)
        query_partner = tl.load(qkv_ptr + qkv_offset + partner, mask=apply_rope, other=0.0).to(
            tl.float32
        )
        key_partner = tl.load(
            qkv_ptr + qkv_offset + hidden_dim + partner,
            mask=apply_rope,
            other=0.0,
        ).to(tl.float32)

        query = tl.where(apply_rope, query * cos + sign * query_partner * sin, query)
        key = tl.where(apply_rope, key * cos + sign * key_partner * sin, key)

        tl.store(
            query_ptr + row * hidden_dim + columns,
            query.to(query_ptr.dtype.element_ty),
        )
        tl.store(
            key_ptr + row * hidden_dim + columns,
            key.to(key_ptr.dtype.element_ty),
        )
        tl.store(value_ptr + row * hidden_dim + columns, value)

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the CUDA container
    HAS_TRITON = False


def layer_norm_modulate(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fused, affine-free LayerNorm followed by AdaLN modulation."""
    batch_size, sequence_length, hidden_dim = hidden.shape
    output = torch.empty_like(hidden)
    _layer_norm_modulate_kernel[(batch_size * sequence_length,)](
        hidden,
        shift,
        scale,
        output,
        sequence_length,
        shift.stride(0),
        eps,
        hidden_dim,
    )
    return output


def layer_norm_modulate_torch(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation used when Triton fusion is not profitable."""
    normalized = F.layer_norm(hidden, (hidden.shape[-1],), eps=eps)
    return torch.addcmul(shift, normalized, scale)


def gate_residual_layer_norm_modulate(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a gated residual update with the next LayerNorm and modulation."""
    batch_size, sequence_length, hidden_dim = hidden.shape
    hidden_output = torch.empty_like(hidden)
    norm_output = torch.empty_like(hidden)
    _gate_residual_layer_norm_modulate_kernel[(batch_size * sequence_length,)](
        hidden,
        gate,
        residual,
        shift,
        scale,
        hidden_output,
        norm_output,
        sequence_length,
        gate.stride(0),
        eps,
        hidden_dim,
    )
    return hidden_output, norm_output


def layer_norm_modulate_packed(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    document_ids: torch.Tensor,
    modulation_stride: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fused LayerNorm and modulation to a packed ragged batch."""
    num_rows, hidden_dim = hidden.shape
    output = torch.empty_like(hidden)
    _layer_norm_modulate_packed_kernel[(num_rows,)](
        hidden,
        shift,
        scale,
        output,
        document_ids,
        modulation_stride,
        eps,
        hidden_dim,
    )
    return output


def gate_residual_layer_norm_modulate_packed(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    document_ids: torch.Tensor,
    modulation_stride: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed-layout variant of :func:`gate_residual_layer_norm_modulate`."""
    num_rows, hidden_dim = hidden.shape
    hidden_output = torch.empty_like(hidden)
    norm_output = torch.empty_like(hidden)
    _gate_residual_layer_norm_modulate_packed_kernel[(num_rows,)](
        hidden,
        gate,
        residual,
        shift,
        scale,
        hidden_output,
        norm_output,
        document_ids,
        modulation_stride,
        eps,
        hidden_dim,
    )
    return hidden_output, norm_output


def split_qkv_apply_rope(
    qkv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV and apply partial RoPE without repacking copies."""
    num_rows = qkv.shape[0] * qkv.shape[1]
    hidden_dim = num_heads * head_dim
    query = torch.empty(num_rows, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    _split_qkv_apply_rope_kernel[(num_rows,)](
        qkv.view(num_rows, 3 * hidden_dim),
        query,
        key,
        value,
        rope_cos,
        rope_sin,
        sequence_length,
        hidden_dim,
        rope_dim,
    )
    return query, key, value


def split_qkv_apply_rope_packed(
    qkv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV and apply partial RoPE to packed documents."""
    num_rows = qkv.shape[0]
    hidden_dim = num_heads * head_dim
    query = torch.empty(num_rows, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    _split_qkv_apply_rope_packed_kernel[(num_rows,)](
        qkv,
        query,
        key,
        value,
        rope_cos,
        rope_sin,
        positions,
        hidden_dim,
        rope_dim,
    )
    return query, key, value
