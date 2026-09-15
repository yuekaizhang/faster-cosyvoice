# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Flow inference built on top of :class:`FlashInferDiT`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from faster_cosyvoice.token2wav.cosyvoice.utils.mask import make_pad_mask


@torch.inference_mode()
def solve_euler_batched(
    decoder: Any,
    noise: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    condition: torch.Tensor,
    num_timesteps: int = 10,
    streaming: bool = False,
) -> torch.Tensor:
    """Run the classifier-free-guidance Euler solver for a batch.

    CosyVoice's original solver preallocates buffers for one sample. This
    implementation constructs the ``2 * batch_size`` conditional/unconditional
    stack required by packed inference.
    """
    batch_size = mu.shape[0]
    time_span = torch.linspace(
        0,
        1,
        num_timesteps + 1,
        device=mu.device,
        dtype=speaker_embeddings.dtype,
    )
    time_span = 1 - torch.cos(time_span * 0.5 * torch.pi)
    time = time_span[0]
    step_size = time_span[1] - time_span[0]

    mask_input = mask.repeat(2, 1, 1).to(speaker_embeddings.dtype)
    mu_input = torch.cat([mu, torch.zeros_like(mu)], dim=0).to(speaker_embeddings.dtype)
    speaker_input = torch.cat([speaker_embeddings, torch.zeros_like(speaker_embeddings)], dim=0)
    condition_input = torch.cat([condition, torch.zeros_like(condition)], dim=0).to(
        speaker_embeddings.dtype
    )
    time_input = torch.zeros(2 * batch_size, device=mu.device, dtype=speaker_embeddings.dtype)

    sample = noise.to(speaker_embeddings.dtype)
    for step in range(1, len(time_span)):
        sample_input = sample.repeat(2, 1, 1)
        time_input.fill_(time)
        derivative = decoder.forward_estimator(
            sample_input,
            mask_input,
            mu_input,
            time_input,
            speaker_input,
            condition_input,
            streaming,
        )
        conditional, unconditional = torch.split(derivative, [batch_size, batch_size], dim=0)
        derivative = (
            1.0 + decoder.inference_cfg_rate
        ) * conditional - decoder.inference_cfg_rate * unconditional
        sample = sample + step_size * derivative
        time = time + step_size
        if step < len(time_span) - 1:
            step_size = time_span[step + 1] - time
    return sample


def _pack_tokens(
    token_list: Sequence[Sequence[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad integer token sequences and return their original lengths."""
    token_lengths = torch.tensor([len(tokens) for tokens in token_list], device=device)
    padded_tokens = torch.zeros(
        len(token_list),
        int(token_lengths.max()),
        dtype=torch.long,
        device=device,
    )
    for index, tokens in enumerate(token_list):
        padded_tokens[index, : len(tokens)] = torch.tensor(tokens, device=device)
    return padded_tokens, token_lengths


def _copy_prompt_features(
    prompt_features: Sequence[torch.Tensor],
    *,
    batch_size: int,
    max_mel_length: int,
    output_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    """Copy variable-length prompt Mel features into a padded condition tensor."""
    condition = torch.zeros(
        batch_size,
        max_mel_length,
        output_size,
        device=device,
        dtype=dtype,
    )
    prompt_lengths = []
    for index, prompt in enumerate(prompt_features):
        prompt_length = prompt.shape[1]
        prompt_lengths.append(prompt_length)
        condition[index, :prompt_length] = prompt[0].to(dtype)
    return condition.transpose(1, 2), prompt_lengths


@torch.inference_mode()
def flow_inference_batched(
    flow: Any,
    token_list: Sequence[Sequence[int]],
    prompt_features: Sequence[torch.Tensor],
    speaker_embeddings: torch.Tensor,
) -> list[torch.Tensor]:
    """Run offline Flow inference for variable-length samples.

    Args:
      flow:
        ``CausalMaskedDiffWithDiT`` whose estimator is ``FlashInferDiT``.
      token_list:
        Prompt and generated speech-token sequences, one per sample.
      prompt_features:
        Prompt Mel tensors with shape ``(1, prompt_length, 80)``.
      speaker_embeddings:
        Speaker embeddings with shape ``(batch_size, 192)``.

    Returns:
      Generated Mel tensors with shape ``(1, 80, generated_length)``.
    """
    device = speaker_embeddings.device
    dtype = next(flow.parameters()).dtype
    batch_size = len(token_list)

    tokens, token_lengths = _pack_tokens(token_list, device)
    speaker_embeddings = F.normalize(speaker_embeddings.to(dtype), dim=1)
    speaker_embeddings = flow.spk_embed_affine_layer(speaker_embeddings)

    token_mask = (~make_pad_mask(token_lengths)).unsqueeze(-1).to(speaker_embeddings)
    token_features = flow.input_embedding(torch.clamp(tokens, min=0)) * token_mask
    mu = flow.pre_lookahead_layer(token_features)
    mu = mu.repeat_interleave(flow.token_mel_ratio, dim=1)

    mel_lengths = token_lengths * flow.token_mel_ratio
    max_mel_length = int(mel_lengths.max())
    condition, prompt_lengths = _copy_prompt_features(
        prompt_features,
        batch_size=batch_size,
        max_mel_length=max_mel_length,
        output_size=flow.output_size,
        device=device,
        dtype=mu.dtype,
    )
    mel_mask = (~make_pad_mask(mel_lengths, max_len=max_mel_length)).to(mu)
    noise = torch.randn(
        batch_size,
        flow.output_size,
        max_mel_length,
        device=device,
        dtype=mu.dtype,
    )

    generated = solve_euler_batched(
        flow.decoder,
        noise,
        mu=mu.transpose(1, 2).contiguous(),
        mask=mel_mask.unsqueeze(1),
        speaker_embeddings=speaker_embeddings,
        condition=condition,
        num_timesteps=10,
    )
    return [
        generated[index : index + 1, :, prompt_lengths[index] : int(mel_lengths[index])].float()
        for index in range(batch_size)
    ]


def _streaming_token_features(
    flow: Any,
    tokens: Sequence[int],
    *,
    finalize: bool,
    device: torch.device,
) -> torch.Tensor:
    """Apply token embedding and right-lookahead handling to one stream."""
    token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    embedded = flow.input_embedding(torch.clamp(token_tensor, min=0))
    if finalize:
        features = flow.pre_lookahead_layer(embedded)
    else:
        lookahead = flow.pre_lookahead_len
        features = flow.pre_lookahead_layer(
            embedded[:, :-lookahead], context=embedded[:, -lookahead:]
        )
    return features.repeat_interleave(flow.token_mel_ratio, dim=1)


@torch.inference_mode()
def flow_inference_batched_streaming(
    flow: Any,
    token_list: Sequence[Sequence[int]],
    prompt_features: Sequence[torch.Tensor],
    speaker_embeddings: torch.Tensor,
    finalize_list: Sequence[bool],
) -> list[torch.Tensor]:
    """Run deterministic full-prefix Flow inference for several streams.

    Non-final streams route the last ``pre_lookahead_len`` tokens to the
    lookahead convolution as context. Each document is embedded separately so
    its result matches single-session streaming, then the DiT work is packed.
    The decoder's fixed noise prefix keeps repeated full-prefix inference
    deterministic across chunks.
    """
    if len(token_list) != len(finalize_list):
        raise ValueError("token_list and finalize_list must have the same length")

    device = speaker_embeddings.device
    dtype = next(flow.parameters()).dtype
    batch_size = len(token_list)

    speaker_embeddings = F.normalize(speaker_embeddings.float(), dim=1)
    speaker_embeddings = flow.spk_embed_affine_layer(speaker_embeddings.to(dtype))

    mu_list = [
        _streaming_token_features(
            flow,
            tokens,
            finalize=finalize_list[index],
            device=device,
        )
        for index, tokens in enumerate(token_list)
    ]
    mel_lengths = [features.shape[1] for features in mu_list]
    max_mel_length = max(mel_lengths)

    mu = torch.zeros(
        batch_size,
        max_mel_length,
        mu_list[0].shape[-1],
        device=device,
        dtype=dtype,
    )
    for index, features in enumerate(mu_list):
        mu[index, : mel_lengths[index]] = features[0]

    condition, prompt_lengths = _copy_prompt_features(
        prompt_features,
        batch_size=batch_size,
        max_mel_length=max_mel_length,
        output_size=flow.output_size,
        device=device,
        dtype=dtype,
    )
    mel_lengths_tensor = torch.tensor(mel_lengths, device=device)
    mel_mask = (~make_pad_mask(mel_lengths_tensor, max_len=max_mel_length)).to(mu)

    noise = flow.decoder.rand_noise[:, :, :max_mel_length].to(device).to(dtype)
    noise = noise.expand(batch_size, -1, -1)
    generated = solve_euler_batched(
        flow.decoder,
        noise,
        mu=mu.transpose(1, 2).contiguous(),
        mask=mel_mask.unsqueeze(1),
        speaker_embeddings=speaker_embeddings,
        condition=condition,
        num_timesteps=10,
        streaming=True,
    )
    return [
        generated[index : index + 1, :, prompt_lengths[index] : mel_lengths[index]].float()
        for index in range(batch_size)
    ]


@torch.inference_mode()
def token2wav_forward_batched(
    model: Any,
    generated_speech_tokens: Sequence[Sequence[int]],
    prompt_audio: Sequence[torch.Tensor],
    prompt_sample_rates: Sequence[int],
) -> list[torch.Tensor]:
    """Run the original offline Token2Wav entry point as a packed batch."""
    if not all(sample_rate == 16000 for sample_rate in prompt_sample_rates):
        raise ValueError("all prompt audio must be sampled at 16 kHz")

    prompt_speech_tokens = model.prompt_audio_tokenization(prompt_audio)
    prompt_mels, prompt_mel_lengths = model.get_prompt_mels(prompt_audio, prompt_sample_rates)
    speaker_embeddings = model.get_spk_emb(prompt_audio).to(model.device)

    token_list = []
    prompt_features = []
    for index, generated_tokens in enumerate(generated_speech_tokens):
        prompt_token_length = min(
            int(prompt_mel_lengths[index].item() / 2),
            len(prompt_speech_tokens[index]),
        )
        prompt_tokens = prompt_speech_tokens[index][:prompt_token_length]
        token_list.append(prompt_tokens + list(generated_tokens))
        prompt_features.append(
            prompt_mels[index : index + 1, : 2 * prompt_token_length].to(model.device)
        )

    generated_mels = flow_inference_batched(
        model.flow, token_list, prompt_features, speaker_embeddings
    )
    waveforms = []
    for mel in generated_mels:
        waveform, _ = model.hift.inference(speech_feat=mel, finalize=True)
        waveforms.append(waveform)
    return waveforms
