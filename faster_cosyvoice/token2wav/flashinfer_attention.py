# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer ragged-attention planning for CosyVoice3 DiT."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import flashinfer
import torch

DEFAULT_WORKSPACE_SIZE = 64 * 1024 * 1024


def build_chunk_causal_mask(
    document_lengths: Sequence[int],
    chunk_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build FlashInfer's flattened chunk-causal mask.

    A query can attend to keys in its own chunk or an earlier chunk.  FlashInfer
    expects each document's row-major mask to be flattened and then concatenated;
    ``True`` means that the query-key pair is visible.
    """
    masks = []
    for length in document_lengths:
        positions = torch.arange(length, device=device)
        query_chunks = positions[:, None] // chunk_size
        key_chunks = positions[None, :] // chunk_size
        masks.append((key_chunks <= query_chunks).flatten())
    return torch.cat(masks)


class RaggedAttentionRunner:
    """Own one FlashInfer wrapper and its currently prepared execution plan.

    A FlashInfer wrapper retains only one plan.  ``FlashInferDiT`` therefore
    pools runners when several sequence shapes recur; this class only caches
    the mask and plan belonging to its current shape.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
        workspace_size: int = DEFAULT_WORKSPACE_SIZE,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)

        self._workspace = torch.zeros(workspace_size, dtype=torch.uint8, device=self.device)
        self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self._workspace, "NHD")
        self._planned_key: tuple[Any, ...] | None = None
        self._mask_key: tuple[tuple[int, ...], int] | None = None
        self._mask: torch.Tensor | None = None

        # Exposed for the lightweight plan-cache tests and profiling.
        self.plan_calls = 0

    def _get_custom_mask(self, document_lengths: Sequence[int], chunk_size: int) -> torch.Tensor:
        key = (tuple(document_lengths), chunk_size)
        if key == self._mask_key:
            assert self._mask is not None
            return self._mask

        if len(set(document_lengths)) == 1:
            # Classifier-free guidance commonly produces identical documents.
            # Build one square block and tile it instead of rebuilding it.
            block = build_chunk_causal_mask(document_lengths[:1], chunk_size, self.device)
            mask = block.repeat(len(document_lengths))
        else:
            mask = build_chunk_causal_mask(document_lengths, chunk_size, self.device)

        self._mask_key = key
        self._mask = mask
        return mask

    def _make_indptr(self, document_lengths: Sequence[int]) -> torch.Tensor:
        lengths = torch.tensor(document_lengths, dtype=torch.int32, device=self.device)
        indptr = torch.zeros(len(document_lengths) + 1, dtype=torch.int32, device=self.device)
        indptr[1:] = torch.cumsum(lengths, dim=0)
        return indptr

    def _prepare(
        self,
        document_lengths: Sequence[int],
        dtype: torch.dtype,
        chunk_size: int | None,
    ) -> None:
        plan_key = (tuple(document_lengths), dtype, chunk_size)
        if plan_key == self._planned_key:
            return

        options: dict[str, Any] = {}
        if chunk_size is not None:
            # Use custom_mask, not packed_custom_mask. FlashInfer 0.6.13 has a
            # byte/element offset mismatch in the latter for multiple documents.
            options["custom_mask"] = self._get_custom_mask(document_lengths, chunk_size)

            # Split-KV scheduling depends on the total batch composition and can
            # make one streaming document's output vary with its neighbours.
            # Disabling it preserves B=1 versus B=N streaming parity.
            options["disable_split_kv"] = True

        indptr = self._make_indptr(document_lengths)
        self.wrapper.plan(
            indptr,
            indptr,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            causal=False,
            sm_scale=self.head_dim**-0.5,
            q_data_type=dtype,
            kv_data_type=dtype,
            **options,
        )
        self._planned_key = plan_key
        self.plan_calls += 1

    def plan(
        self,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> None:
        """Prepare equal-length documents."""
        self._prepare([sequence_length] * batch_size, dtype, chunk_size)

    def plan_docs(
        self,
        document_lengths: Sequence[int],
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> None:
        """Prepare a variable-length packed batch."""
        self._prepare(document_lengths, dtype, chunk_size)
