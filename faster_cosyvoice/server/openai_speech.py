# faster_cosyvoice/server/openai_speech.py
"""每请求流水线（spec §6.2）：resolve voice → prompt → AsyncLLM DELTA 流
→ ChunkPlanner → batcher → PCM 增量。"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

import torch

from faster_cosyvoice.llm.engine import (make_stream_sampling_params,
                                         stream_token_ids)
from faster_cosyvoice.llm.prompt import build_prompt
from faster_cosyvoice.server.audio_encode import pcm16_bytes, wav_stream_header
from faster_cosyvoice.server.protocol import SpeechRequest, decode_ref_audio
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession

logger = logging.getLogger(__name__)
SAMPLE_RATE = 24000


class NoSpeechTokens(RuntimeError):
    """LLM 结束且 0 个有效 speech token（spec §7 → 500）。"""


async def resolve_condition(state, req: SpeechRequest):
    """→ (RefCondition, ref_text)。voice 命中注册表；否则解析 ref_audio。
    解码/前端为阻塞 CPU 工作，丢 executor 避免卡事件循环。"""
    if req.voice is not None:
        entry = state.voices.get(req.voice)
        if entry is None:
            raise KeyError(f"voice 未注册: {req.voice}")
        return entry

    def _work():
        wav, sr = decode_ref_audio(req.ref_audio,
                                   state.server_cfg.max_ref_seconds)
        return state.frontend.process(torch.from_numpy(wav), sr, req.ref_text)

    cond = await asyncio.get_running_loop().run_in_executor(None, _work)
    return cond, req.ref_text


async def synthesize_pcm(state, req: SpeechRequest,
                         cond_pair=None) -> AsyncGenerator[bytes, None]:
    """yield 原始 PCM_16 chunk（不含 WAV 头）。取消即自动 abort LLM 请求。
    cond_pair=(RefCondition, ref_text)：流式路径由 handler 在响应头发出前
    预先 resolve（坏 voice/ref 才能返回 400）；None 则内部解析。"""
    t_start = time.perf_counter()
    cond, ref_text = (cond_pair if cond_pair is not None
                      else await resolve_condition(state, req))
    prompt = build_prompt(state.tokenizer, ref_text, req.input,
                          cond.prompt_tokens_llm)
    text_len = len(state.tokenizer.encode(req.input))
    sp = make_stream_sampling_params(state.llm_cfg, state.codec,
                                     text_len, req.seed)
    # [M3.5-r2] chunk 参数来自 ServerConfig（默认 15/×2 = 现行行为；
    # uniform-25 = frames=25, scale=1）。max_hop 走默认 4×chunk_size —
    # scale=1 时 hop 恒为 chunk_size，封顶不生效。
    session = StreamSession(
        cond=cond, planner=ChunkPlanner(
            len(cond.prompt_tokens_flow),
            chunk_size=state.server_cfg.codec_chunk_frames,
            scale=state.server_cfg.codec_chunk_scale))
    request_id = str(uuid.uuid4())
    ttfa_ms = None

    async def flush(finished: bool):
        nonlocal ttfa_ms
        while True:
            plan = session.planner.next_chunk(len(session.tokens), finished)
            if plan is None:
                return
            pcm = await state.batcher.submit(session, plan,
                                             chunk_index=session.chunk_index)
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t_start) * 1000
            yield pcm16_bytes(pcm)
            if plan.finalize:
                return

    async for delta, finished in stream_token_ids(
            state.engine, prompt, sp, request_id):
        session.tokens.extend(state.codec.extract(delta))
        async for chunk in flush(finished=False):
            yield chunk
    async for chunk in flush(finished=True):
        yield chunk

    if not session.tokens:
        raise NoSpeechTokens("LLM 未产出有效 speech token")
    logger.info(json.dumps(dict(
        event="request_done", request_id=request_id,
        ttfa_ms=round(ttfa_ms or -1, 1),
        chunks=session.chunk_index, tokens=len(session.tokens),
        audio_s=round(session.speech_offset / SAMPLE_RATE, 2),
        wall_s=round(time.perf_counter() - t_start, 2)), ensure_ascii=False))


async def synthesize_response_chunks(state, req,
                                     cond_pair=None) -> AsyncGenerator[bytes, None]:
    """流式响应体：wav 先发未知长度头，再全是 PCM。"""
    if req.response_format == "wav":
        yield wav_stream_header(SAMPLE_RATE)
    async for chunk in synthesize_pcm(state, req, cond_pair):
        yield chunk
