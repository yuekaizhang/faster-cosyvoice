"""Per-request pipeline from voice conditioning to streamed PCM.

The path is: resolve reference voice, build the LLM prompt, consume AsyncLLM
token deltas, plan audio chunks, submit them to Token2Wav, and emit PCM.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

import torch

from faster_cosyvoice.llm.engine import make_stream_sampling_params, stream_token_ids
from faster_cosyvoice.llm.prompt import build_prompt
from faster_cosyvoice.server.audio_encode import pcm16_bytes, wav_stream_header
from faster_cosyvoice.server.leading_silence import LeadingSilenceTrimmer
from faster_cosyvoice.server.protocol import SpeechRequest, decode_ref_audio
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession

logger = logging.getLogger(__name__)
SAMPLE_RATE = 24000


class NoSpeechTokens(RuntimeError):
    """The LLM completed without producing any valid speech token."""


async def resolve_condition(state, req: SpeechRequest):
    """Resolve a registered voice or process request-level reference audio.

    Audio decoding and frontend work are blocking, so request-level references
    run in an executor instead of blocking the event loop.
    """
    if req.voice is not None:
        entry = state.voices.get(req.voice)
        if entry is None:
            raise KeyError(f"voice is not registered: {req.voice}")
        return entry

    def _work():
        wav, sr = decode_ref_audio(req.ref_audio,
                                   state.server_cfg.max_ref_seconds)
        return state.frontend.process(torch.from_numpy(wav), sr, req.ref_text)

    cond = await asyncio.get_running_loop().run_in_executor(None, _work)
    return cond, req.ref_text


async def synthesize_pcm(state, req: SpeechRequest,
                         cond_pair=None) -> AsyncGenerator[bytes, None]:
    """Yield raw PCM16 chunks without a WAV header.

    Streaming handlers resolve the condition before sending response headers so
    invalid voice input can still return HTTP 400.  Closing this generator on a
    disconnect also closes the LLM token stream and aborts that request.
    """
    t_start = time.perf_counter()
    condition, ref_text = (
        cond_pair if cond_pair is not None else await resolve_condition(state, req)
    )
    prompt = build_prompt(state.tokenizer, ref_text, req.input,
                          condition.prompt_tokens_llm)
    text_len = len(state.tokenizer.encode(req.input))
    sampling_params = make_stream_sampling_params(
        state.llm_cfg, state.codec, text_len, req.seed
    )
    # A growth factor of one keeps every hop at the configured chunk size;
    # larger factors grow until ChunkPlanner's default 4x cap.
    session = StreamSession(
        cond=condition, planner=ChunkPlanner(
            len(condition.prompt_tokens_flow),
            chunk_size=state.server_cfg.speech_token_chunk_size,
            scale=state.server_cfg.speech_token_chunk_growth))
    request_id = str(uuid.uuid4())
    ttfa_ms = None
    trimmer = (LeadingSilenceTrimmer(
        sample_rate=SAMPLE_RATE,
        preroll_ms=state.server_cfg.leading_silence_preroll_ms,
        max_trim_ms=state.server_cfg.leading_silence_max_ms,
        min_buffer_ms=state.server_cfg.leading_silence_min_buffer_ms)
        if state.server_cfg.trim_leading_silence else None)

    async def emit_ready_chunks(finished: bool):
        nonlocal ttfa_ms
        while True:
            plan = session.planner.next_chunk(len(session.tokens), finished)
            if plan is None:
                return
            pcm = await state.batcher.submit(session, plan,
                                             chunk_index=session.chunk_index)
            payload = pcm16_bytes(pcm)
            if trimmer is not None:
                payload = trimmer.feed(payload, final=plan.finalize)
            if not payload:
                if plan.finalize:
                    return
                continue
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t_start) * 1000
            # Publish playback credit at the route boundary, not when GPU work
            # merely completes.  The deadline-aware token2wav scheduler then
            # knows how much real-time audio this stream has left.
            session.mark_pcm_routed(len(payload) // 2, SAMPLE_RATE,
                                    time.monotonic())
            yield payload
            if plan.finalize:
                return

    async for delta, _finished in stream_token_ids(
            state.engine, prompt, sampling_params, request_id):
        session.tokens.extend(state.codec.extract(delta))
        async for chunk in emit_ready_chunks(finished=False):
            yield chunk
    async for chunk in emit_ready_chunks(finished=True):
        yield chunk

    if not session.tokens:
        raise NoSpeechTokens("LLM produced no valid speech tokens")
    logger.info(json.dumps(dict(
        event="request_done", request_id=request_id,
        ttfa_ms=round(ttfa_ms or -1, 1),
        chunks=session.chunk_index, tokens=len(session.tokens),
        audio_s=round(session.speech_offset / SAMPLE_RATE, 2),
        routed_audio_s=round(session.emitted_duration_s, 2),
        wall_s=round(time.perf_counter() - t_start, 2)), ensure_ascii=False))


async def synthesize_response_chunks(state, req,
                                     cond_pair=None) -> AsyncGenerator[bytes, None]:
    """Yield an unknown-length WAV header when requested, then raw PCM."""
    if req.response_format == "wav":
        yield wav_stream_header(SAMPLE_RATE)
    async for chunk in synthesize_pcm(state, req, cond_pair):
        yield chunk
