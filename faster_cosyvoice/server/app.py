# faster_cosyvoice/server/app.py
"""FastAPI server: validate, load assets, warm up, then serve traffic.

Run with ``uv run faster-cosyvoice-server --port 8000``.
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


def build_app(llm_cfg: LLMConfig, token2wav_cfg: Token2WavConfig,
              server_cfg: ServerConfig) -> FastAPI:

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        state.ready = False
        problems = check_environment(
            # FlashInfer is the optimized streaming Flow path; Torch is fallback.
            require_flashinfer=(token2wav_cfg.estimator_mode == "flashinfer"),
            require_draft_mirror=(llm_cfg.draft_model is not None
                                  and llm_cfg.repetition_penalty != 1.0))
        if problems:
            raise RuntimeError("环境自检失败：" + "; ".join(problems))
        model_dir = ensure_token2wav_assets(token2wav_cfg.model_dir)
        from transformers import AutoTokenizer
        state.llm_cfg = llm_cfg
        state.server_cfg = server_cfg
        state.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.target_model)
        state.codec = SpeechTokenCodec(state.tokenizer)
        state.engine = create_async_llm(llm_cfg)
        state.frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx",
                                          device=token2wav_cfg.device,
                                          cache_size=server_cfg.voice_cache_size,
                                          campplus_trt=(
                                              token2wav_cfg
                                              .speaker_encoder_tensorrt))
        stream_buckets = ([int(v) for v in
                           token2wav_cfg.streaming_flow_graph_buckets.split(",")]
                          if token2wav_cfg.streaming_flow_graph_buckets else None)
        hift_buckets = ([int(v) for v in
                         token2wav_cfg.streaming_vocoder_graph_buckets.split(",")]
                        if token2wav_cfg.streaming_vocoder_graph_buckets else None)
        state.token2wav = CosyVoice3Token2Wav(
            model_dir, device=token2wav_cfg.device,
            estimator_mode=token2wav_cfg.estimator_mode,
            hift_compile=token2wav_cfg.vocoder_compile,
            stream_graph_buckets=stream_buckets,
            hift_graph_buckets=hift_buckets)
        state.batcher = Token2WavWorker(state.token2wav,
                                        mode=token2wav_cfg.batch_mode,
                                        max_batch=token2wav_cfg.batch_size,
                                        scheduler_mode=(
                                            token2wav_cfg.scheduler_mode),
                                        deadline_reserve_s=(
                                            token2wav_cfg.deadline_reserve_s))
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the public CLI.

    Long, descriptive option names are canonical.  The earlier abbreviated
    names remain as hidden aliases so existing deployment commands continue to
    work while ``--help`` stays readable.
    """
    p = argparse.ArgumentParser(
        description="Serve CosyVoice3 through an OpenAI-compatible speech API.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--target-model",
                   default="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF")
    p.add_argument("--draft-model", default="yuekai/cosyvoice3_llm_dspark")
    p.add_argument("--token2wav-dir", default="models/Fun-CosyVoice3-0.5B-2512")
    p.add_argument("--token2wav-device", default="cuda:0")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--request-timeout-s", type=float, default=300.0,
                   help="Timeout for non-streaming requests; streaming timeouts "
                        "are controlled by the client (default: 300).")
    p.add_argument("--streaming-flow-estimator", default="flashinfer",
                   choices=["torch", "flashinfer"],
                   help="Flow estimator used for streaming synthesis "
                        "(default: flashinfer; torch is the fallback).")
    p.add_argument("--stream-estimator", dest="streaming_flow_estimator",
                   choices=["torch", "flashinfer"], default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--token2wav-batch-mode", default="packed",
                   choices=["serial", "packed"],
                   help="Cross-session token2wav execution mode "
                        "(default: packed; requires FlashInfer).")
    p.add_argument("--t2w-batch-mode", dest="token2wav_batch_mode",
                   choices=["serial", "packed"], default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--token2wav-batch-size", type=int, default=8,
                   help="Maximum packed token2wav batch size (default: 8).")
    p.add_argument("--t2w-batch-size", dest="token2wav_batch_size", type=int,
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--token2wav-deadline-reserve-ms", type=float, default=100.0,
                   help="Prioritize an established stream when its playback "
                        "buffer is this close to empty (default: 100 ms).")
    p.add_argument("--t2w-deadline-reserve-ms",
                   dest="token2wav_deadline_reserve_ms", type=float,
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--token2wav-scheduler", default="deadline",
                   choices=["legacy", "deadline"],
                   help="token2wav scheduling policy (default: deadline; "
                        "legacy is retained for compatibility experiments).")
    p.add_argument("--t2w-scheduler", dest="token2wav_scheduler",
                   choices=["legacy", "deadline"], default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--vocoder-compile", action="store_true",
                   help="Compile the HiFT vocoder and quantize offline input "
                        "lengths to 64-Mel-frame buckets. Adds 15-20 seconds "
                        "of one-time startup warmup.")
    p.add_argument("--hift-compile", dest="vocoder_compile", action="store_true",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--streaming-flow-graph-buckets", default=None,
                   help="Comma-separated streaming Flow sequence-length "
                        "buckets in Mel frames, for example "
                        "512,640,768,896,1024,1280. Single-session only.")
    p.add_argument("--stream-graph-buckets",
                   dest="streaming_flow_graph_buckets",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--streaming-vocoder-graph-buckets", default=None,
                   help="Comma-separated streaming vocoder input buckets in "
                        "Mel frames, for example 64,128,192,256,384,512. "
                        "Applies to non-final chunks only.")
    p.add_argument("--hift-graph-buckets",
                   dest="streaming_vocoder_graph_buckets",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--speech-token-chunk-size", type=int, default=15,
                   help="Speech tokens consumed by the first streaming hop "
                        "(default: 15; use 25 for fixed-shape graph mode).")
    p.add_argument("--codec-chunk-frames", dest="speech_token_chunk_size",
                   type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--speech-token-chunk-growth", type=int, default=2,
                   help="Multiplier applied to successive streaming hops "
                        "(default: 2; use 1 for fixed-size hops).")
    p.add_argument("--codec-chunk-scale", dest="speech_token_chunk_growth",
                   type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--trim-leading-silence", action="store_true",
                   help="Trim bounded leading silence while retaining pre-roll. "
                        "This changes only the beginning of the PCM stream.")
    p.add_argument("--leading-silence-preroll-ms", type=float, default=20.0,
                   help="Audio retained before detected speech (default: 20 ms).")
    p.add_argument("--leading-silence-max-ms", type=float, default=2000.0,
                   help="Maximum leading-silence window (default: 2000 ms).")
    p.add_argument("--leading-silence-min-buffer-ms", type=float, default=400.0,
                   help="Playable audio buffered before the first trimmed "
                        "response chunk (default: 400 ms).")
    p.add_argument("--speaker-encoder-tensorrt", action="store_true",
                   help="Run the CampPlus speaker encoder with TensorRT "
                        "instead of ONNX Runtime CPU. The first run builds and "
                        "caches an engine.")
    p.add_argument("--campplus-trt", dest="speaker_encoder_tensorrt",
                   action="store_true", default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    return p


def main():
    args = build_argument_parser().parse_args()
    if (args.token2wav_batch_mode == "packed"
            and args.streaming_flow_estimator != "flashinfer"):
        raise SystemExit("--token2wav-batch-mode packed requires "
                         "--streaming-flow-estimator flashinfer; use "
                         "--token2wav-batch-mode serial with the torch estimator")
    if args.token2wav_batch_size < 1:
        raise SystemExit("--token2wav-batch-size must be >= 1")
    if args.token2wav_deadline_reserve_ms < 0:
        raise SystemExit("--token2wav-deadline-reserve-ms must be >= 0")
    if (args.leading_silence_preroll_ms < 0
            or args.leading_silence_max_ms < 0
            or args.leading_silence_min_buffer_ms < 0
            or args.leading_silence_preroll_ms > args.leading_silence_max_ms):
        raise SystemExit("require 0 <= --leading-silence-preroll-ms "
                         "<= --leading-silence-max-ms")

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # 同 offline 的 fork segfault 规避
    draft = None if args.draft_model in (None, "none") else args.draft_model
    llm_cfg = LLMConfig(target_model=args.target_model, draft_model=draft,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    token2wav_cfg = Token2WavConfig(
        model_dir=args.token2wav_dir,
        device=args.token2wav_device,
        estimator_mode=args.streaming_flow_estimator,
        batch_mode=args.token2wav_batch_mode,
        batch_size=args.token2wav_batch_size,
        scheduler_mode=args.token2wav_scheduler,
        deadline_reserve_s=args.token2wav_deadline_reserve_ms / 1000,
        vocoder_compile=args.vocoder_compile,
        speaker_encoder_tensorrt=args.speaker_encoder_tensorrt,
        streaming_flow_graph_buckets=args.streaming_flow_graph_buckets,
        streaming_vocoder_graph_buckets=args.streaming_vocoder_graph_buckets,
    )
    server_cfg = ServerConfig(host=args.host, port=args.port,
                              gpu_memory_utilization=args.gpu_memory_utilization,
                              request_timeout_s=args.request_timeout_s,
                              speech_token_chunk_size=args.speech_token_chunk_size,
                              speech_token_chunk_growth=(
                                  args.speech_token_chunk_growth),
                              trim_leading_silence=args.trim_leading_silence,
                              leading_silence_preroll_ms=(
                                  args.leading_silence_preroll_ms),
                              leading_silence_max_ms=(
                                  args.leading_silence_max_ms),
                              leading_silence_min_buffer_ms=(
                                  args.leading_silence_min_buffer_ms))
    uvicorn.run(build_app(llm_cfg, token2wav_cfg, server_cfg),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
