# faster_cosyvoice/server/app.py
"""FastAPI server（spec §5.4/§5.5）。启动：自检→资产→引擎→warmup→接流量。

用法：python -m faster_cosyvoice.server.app --port 8000 [--draft-model none]
"""
import argparse
import asyncio
import contextlib
import io
import logging
from types import SimpleNamespace

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import LLMConfig, ServerConfig, Token2WavConfig
from faster_cosyvoice.envcheck import check_environment
from faster_cosyvoice.llm.engine import create_async_llm
from faster_cosyvoice.llm.tokens import SpeechTokenCodec
from faster_cosyvoice.server.openai_speech import (NoSpeechTokens, SAMPLE_RATE,
                                                   resolve_condition,
                                                   synthesize_pcm,
                                                   synthesize_response_chunks)
from faster_cosyvoice.server.protocol import (SpeechRequest, VoiceRequest,
                                              decode_ref_audio)
from faster_cosyvoice.streaming.batcher import Token2WavWorker
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
state = SimpleNamespace()


def build_app(llm_cfg: LLMConfig, t2w_cfg: Token2WavConfig,
              server_cfg: ServerConfig) -> FastAPI:

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        problems = check_environment(
            # [M3] 流式 estimator 可选 flashinfer（--stream-estimator）
            require_flashinfer=(t2w_cfg.estimator_mode == "flashinfer"),
            require_draft_mirror=(llm_cfg.draft_model is not None
                                  and llm_cfg.repetition_penalty != 1.0))
        if problems:
            raise RuntimeError("环境自检失败：" + "; ".join(problems))
        model_dir = ensure_token2wav_assets(t2w_cfg.model_dir)
        from transformers import AutoTokenizer
        state.llm_cfg = llm_cfg
        state.server_cfg = server_cfg
        state.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.target_model)
        state.codec = SpeechTokenCodec(state.tokenizer)
        state.engine = create_async_llm(llm_cfg)
        state.frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx",
                                          device=t2w_cfg.device,
                                          cache_size=server_cfg.voice_cache_size,
                                          campplus_trt=t2w_cfg.campplus_trt)
        state.token2wav = CosyVoice3Token2Wav(
            model_dir, device=t2w_cfg.device,
            estimator_mode=t2w_cfg.estimator_mode,
            hift_compile=t2w_cfg.hift_compile)
        state.batcher = Token2WavWorker(state.token2wav,
                                        mode=t2w_cfg.batch_mode,
                                        max_batch=t2w_cfg.batch_size)
        state.voices = {}
        await state.batcher.start()
        await _warmup()
        logger.info("server ready")
        yield
        await state.batcher.stop()
        state.engine.shutdown()

    async def _warmup():
        """一条 dummy 请求打通全链路（spec §7：warmup 失败即启动失败）。"""
        import base64
        buf = io.BytesIO()
        sf.write(buf, torch.randn(16000).mul(0.05).numpy(), 16000,
                 format="WAV")
        req = SpeechRequest(
            input="你好。", stream=True, response_format="pcm",
            ref_audio="data:audio/wav;base64,"
                      + base64.b64encode(buf.getvalue()).decode(),
            ref_text="测试。")
        n = 0
        async for _ in synthesize_pcm(state, req):
            n += 1
        if n == 0:
            raise RuntimeError("warmup 未产出音频")
        logger.info("warmup OK (%d chunks)", n)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        """流式路径先 resolve（坏 voice/ref → 400）；响应头发出后
        生成中途出错只能截断流（chunked transfer 固有限制）。"""
        media = "audio/wav" if req.response_format == "wav" else "audio/pcm"
        try:
            if req.stream:
                cond_pair = await resolve_condition(state, req)
                return StreamingResponse(
                    synthesize_response_chunks(state, req, cond_pair),
                    media_type=media)
            chunks = []
            async with asyncio.timeout(state.server_cfg.request_timeout_s):
                async for c in synthesize_pcm(state, req):
                    chunks.append(c)
            pcm = b"".join(chunks)
            if req.response_format == "pcm":
                return Response(content=pcm, media_type=media)
            import numpy as np
            buf = io.BytesIO()
            sf.write(buf, np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE,
                     format="WAV")
            return Response(content=buf.getvalue(), media_type=media)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        except NoSpeechTokens as e:
            raise HTTPException(500, str(e))
        except TimeoutError:
            raise HTTPException(504, "请求超时")

    @app.post("/v1/audio/voices")
    async def register_voice(req: VoiceRequest):
        def _work():
            wav, sr = decode_ref_audio(req.ref_audio,
                                       state.server_cfg.max_ref_seconds)
            return state.frontend.process(torch.from_numpy(wav), sr,
                                          req.ref_text)
        try:
            cond = await asyncio.get_running_loop().run_in_executor(None,
                                                                    _work)
        except ValueError as e:
            raise HTTPException(400, str(e))
        state.voices[req.name] = (cond, req.ref_text)
        return {"success": True, "voice": req.name}

    @app.get("/v1/audio/voices")
    async def list_voices():
        return {"voices": sorted(state.voices)}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--target-model",
                   default="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF")
    p.add_argument("--draft-model", default="yuekai/cosyvoice3_llm_dspark")
    p.add_argument("--token2wav-dir", default="models/Fun-CosyVoice3-0.5B-2512")
    p.add_argument("--token2wav-device", default="cuda:0")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--request-timeout-s", type=float, default=300.0,
                   help="非流式请求超时（spec §7；流式超时靠客户端）")
    p.add_argument("--stream-estimator", default="flashinfer",
                   choices=["torch", "flashinfer"],
                   help="[M3] 流式 flow estimator（默认 flashinfer = chunk-causal "
                        "mask fp16 路径；torch 为回退）")
    p.add_argument("--t2w-batch-mode", default="packed",
                   choices=["serial", "packed"],
                   help="[M3] token2wav 批量模式（默认 packed = 跨 session "
                        "flashinfer 批量；要求 --stream-estimator flashinfer）")
    p.add_argument("--hift-compile", action="store_true",
                   help="hift.decode 走 torch.compile + pad-to-bucket（fresh "
                        "shape ~52ms → ~13-21ms；启动一次性 warmup ~15-20s）。"
                        "注意：流式路径同走 compiled decode，波形与 eager 非逐位"
                        "一致（worst-chunk ~6e-2，ASR CER 门通过）")
    p.add_argument("--campplus-trt", action="store_true",
                   help="campplus 说话人 embedding 走 TensorRT（冷 ref resolve "
                        "88.6→23.3ms，spk_emb ~58→~7ms）。首启无 plan 缓存时 "
                        "一次性 build ~2-3min；需要 tensorrt python 包")
    args = p.parse_args()
    if args.t2w_batch_mode == "packed" and args.stream_estimator != "flashinfer":
        raise SystemExit("--t2w-batch-mode packed 要求 --stream-estimator "
                         "flashinfer（torch estimator 无 packed 批量路径）；"
                         "--stream-estimator torch 需同时指定 "
                         "--t2w-batch-mode serial")

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # 同 offline 的 fork segfault 规避
    draft = None if args.draft_model in (None, "none") else args.draft_model
    llm_cfg = LLMConfig(target_model=args.target_model, draft_model=draft,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    t2w_cfg = Token2WavConfig(model_dir=args.token2wav_dir,
                              device=args.token2wav_device,
                              estimator_mode=args.stream_estimator,
                              batch_mode=args.t2w_batch_mode,
                              hift_compile=args.hift_compile,
                              campplus_trt=args.campplus_trt)
    server_cfg = ServerConfig(host=args.host, port=args.port,
                              gpu_memory_utilization=args.gpu_memory_utilization,
                              request_timeout_s=args.request_timeout_s)
    uvicorn.run(build_app(llm_cfg, t2w_cfg, server_cfg),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
