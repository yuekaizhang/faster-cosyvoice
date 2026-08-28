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
from faster_cosyvoice.server.openai_speech import (
    SAMPLE_RATE,
    NoSpeechTokens,
    resolve_condition,
    synthesize_pcm,
    synthesize_response_chunks,
)
from faster_cosyvoice.server.protocol import SpeechRequest, VoiceRequest, decode_ref_audio
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
        state.ready = False
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
        stream_buckets = ([int(v) for v in
                           t2w_cfg.stream_graph_buckets.split(",")]
                          if t2w_cfg.stream_graph_buckets else None)
        hift_buckets = ([int(v) for v in
                         t2w_cfg.hift_graph_buckets.split(",")]
                        if t2w_cfg.hift_graph_buckets else None)
        state.token2wav = CosyVoice3Token2Wav(
            model_dir, device=t2w_cfg.device,
            estimator_mode=t2w_cfg.estimator_mode,
            hift_compile=t2w_cfg.hift_compile,
            stream_graph_buckets=stream_buckets,
            hift_graph_buckets=hift_buckets)
        state.batcher = Token2WavWorker(state.token2wav,
                                        mode=t2w_cfg.batch_mode,
                                        max_batch=t2w_cfg.batch_size,
                                        deadline_reserve_s=(
                                            t2w_cfg.deadline_reserve_s))
        state.voices = {}
        await state.batcher.start()
        await _warmup()
        state.ready = True
        logger.info("server ready")
        yield
        state.ready = False
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

    @app.get("/ready")
    async def ready():
        if not getattr(state, "ready", False):
            raise HTTPException(503, "model is not ready")
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": llm_cfg.target_model,
                "object": "model",
                "owned_by": "faster-cosyvoice",
            }],
        }

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
            raise HTTPException(400, str(e)) from e
        except NoSpeechTokens as e:
            raise HTTPException(500, str(e)) from e
        except TimeoutError as e:
            raise HTTPException(504, "请求超时") from e

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
            raise HTTPException(400, str(e)) from e
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
    p.add_argument("--t2w-batch-size", type=int, default=8,
                   help="packed token2wav 最大 batch（默认 8）")
    p.add_argument("--t2w-deadline-reserve-ms", type=float, default=100.0,
                   help="既有音频流在 playback deadline 前多少毫秒抢占首块工作"
                        "（默认 100，Nari-style deadline-aware scheduling）")
    p.add_argument("--hift-compile", action="store_true",
                   help="hift.decode 走 torch.compile + pad-to-bucket（fresh "
                        "shape ~52ms → ~13-21ms；启动一次性 warmup ~15-20s）。"
                        "注意：流式路径同走 compiled decode，波形与 eager 非逐位"
                        "一致（worst-chunk ~6e-2，ASR CER 门通过）")
    p.add_argument("--stream-graph-buckets", default=None,
                   help="[M3.5-r2] 流式 bucketed CUDA graphs：逗号分隔 mel 帧数"
                        "（如 \"512,640,768,896,1024,1280\"）。仅单 session "
                        "流式命中；dense-SDPA graph 与 eager 非逐位一致"
                        "（ASR CER 门为准）。每 bucket 首遇 capture ~100ms；"
                        "warmup 会预热 warmup voice 命中的桶。建议配合 "
                        "--codec-chunk-frames 25 --codec-chunk-scale 1 "
                        "（chunk 形状可枚举）")
    p.add_argument("--hift-graph-buckets", default=None,
                   help="[M3.5-r4] 流式 hift bucketed CUDA graphs：逗号分隔 "
                        "mel 帧数（如 \"64,128,192,256,384,512\"）。"
                        "finalize=False 中间 chunk 整段 hift 捕成 graph"
                        "（pad-to-bucket，数学等价 ~3e-4；最终 chunk 走原路径）。"
                        "同卡 vLLM 抢占下 chunk-1 hift 16.8→8.9ms、TTFP "
                        "~112→~105ms；per-bucket 首遇 lazy capture。"
                        "质量门 = ASR CER")
    p.add_argument("--codec-chunk-frames", type=int, default=15,
                   help="[M3.5-r2] ChunkPlanner chunk_size（token 数；默认 15 "
                        "= 现行行为；uniform-25 模式设 25）")
    p.add_argument("--codec-chunk-scale", type=int, default=2,
                   help="[M3.5-r2] hop 逐块放大倍率（默认 2 = 现行 ×2 growth；"
                        "1 = uniform hop，chunk 形状可枚举）")
    p.add_argument("--trim-leading-silence", action="store_true",
                   help="按 Nari audible-onset 规则抑制首段静音，保留 pre-roll；"
                        "仅影响开头 PCM，默认关闭")
    p.add_argument("--leading-silence-preroll-ms", type=float, default=20.0,
                   help="首段静音裁剪后保留的 pre-roll（默认 20ms）")
    p.add_argument("--leading-silence-max-ms", type=float, default=2000.0,
                   help="最多等待/裁剪的首段静音窗口（默认 2000ms）")
    p.add_argument("--leading-silence-min-buffer-ms", type=float, default=400.0,
                   help="裁剪后首次发送至少累计的可播放音频（默认 400ms，"
                        "用于避免首包过短后立即 underrun）")
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
    if args.t2w_batch_size < 1:
        raise SystemExit("--t2w-batch-size 必须 >= 1")
    if args.t2w_deadline_reserve_ms < 0:
        raise SystemExit("--t2w-deadline-reserve-ms 必须 >= 0")
    if (args.leading_silence_preroll_ms < 0
            or args.leading_silence_max_ms < 0
            or args.leading_silence_min_buffer_ms < 0
            or args.leading_silence_preroll_ms > args.leading_silence_max_ms):
        raise SystemExit("需满足 0 <= leading-silence-preroll-ms "
                         "<= leading-silence-max-ms")

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # 同 offline 的 fork segfault 规避
    draft = None if args.draft_model in (None, "none") else args.draft_model
    llm_cfg = LLMConfig(target_model=args.target_model, draft_model=draft,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    t2w_cfg = Token2WavConfig(model_dir=args.token2wav_dir,
                              device=args.token2wav_device,
                              estimator_mode=args.stream_estimator,
                              batch_mode=args.t2w_batch_mode,
                              batch_size=args.t2w_batch_size,
                              deadline_reserve_s=(
                                  args.t2w_deadline_reserve_ms / 1000),
                              hift_compile=args.hift_compile,
                              campplus_trt=args.campplus_trt,
                              stream_graph_buckets=args.stream_graph_buckets,
                              hift_graph_buckets=args.hift_graph_buckets)
    server_cfg = ServerConfig(host=args.host, port=args.port,
                              gpu_memory_utilization=args.gpu_memory_utilization,
                              request_timeout_s=args.request_timeout_s,
                              codec_chunk_frames=args.codec_chunk_frames,
                              codec_chunk_scale=args.codec_chunk_scale,
                              trim_leading_silence=args.trim_leading_silence,
                              leading_silence_preroll_ms=(
                                  args.leading_silence_preroll_ms),
                              leading_silence_max_ms=(
                                  args.leading_silence_max_ms),
                              leading_silence_min_buffer_ms=(
                                  args.leading_silence_min_buffer_ms))
    uvicorn.run(build_app(llm_cfg, t2w_cfg, server_cfg),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
