"""FastAPI server: validate, load assets, warm up, then serve traffic.

Run with ``uv run faster-cosyvoice-server --port 8000``.
"""
import asyncio
import contextlib
import io
import logging
from dataclasses import dataclass, field
from typing import Any

import soundfile as sf
import torch
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
from faster_cosyvoice.streaming.token2wav_batcher import Token2WavWorker
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerRuntime:
    """Objects shared by request handlers after lifespan initialization."""

    llm_cfg: LLMConfig
    server_cfg: ServerConfig
    ready: bool = False
    tokenizer: Any = None
    codec: Any = None
    engine: Any = None
    frontend: Any = None
    token2wav: Any = None
    batcher: Any = None
    voices: dict[str, tuple[Any, str]] = field(default_factory=dict)


def build_app(llm_cfg: LLMConfig, token2wav_cfg: Token2WavConfig,
              server_cfg: ServerConfig) -> FastAPI:
    runtime = ServerRuntime(llm_cfg=llm_cfg, server_cfg=server_cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.ready = False
        problems = check_environment(
            # FlashInfer is the optimized streaming Flow path; Torch is fallback.
            require_flashinfer=(token2wav_cfg.estimator_mode == "flashinfer"),
            require_draft_mirror=(llm_cfg.draft_model is not None
                                  and llm_cfg.repetition_penalty != 1.0))
        if problems:
            raise RuntimeError("environment check failed: " + "; ".join(problems))
        model_dir = ensure_token2wav_assets(token2wav_cfg.model_dir)
        from transformers import AutoTokenizer
        runtime.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.target_model)
        runtime.codec = SpeechTokenCodec(runtime.tokenizer)
        runtime.engine = create_async_llm(llm_cfg)
        runtime.frontend = RefAudioFrontend(
            f"{model_dir}/campplus.onnx",
            device=token2wav_cfg.device,
            cache_size=server_cfg.voice_cache_size,
            campplus_trt=token2wav_cfg.speaker_encoder_tensorrt,
        )
        runtime.token2wav = CosyVoice3Token2Wav(
            model_dir, device=token2wav_cfg.device,
            estimator_mode=token2wav_cfg.estimator_mode,
            hift_compile=token2wav_cfg.vocoder_compile,
            stream_graph_buckets=token2wav_cfg.streaming_flow_graph_buckets,
            hift_graph_buckets=token2wav_cfg.streaming_vocoder_graph_buckets)
        runtime.batcher = Token2WavWorker(
            runtime.token2wav,
            mode=token2wav_cfg.batch_mode,
            max_batch=token2wav_cfg.batch_size,
            scheduler_mode=token2wav_cfg.scheduler_mode,
            deadline_reserve_s=token2wav_cfg.deadline_reserve_s,
        )
        await runtime.batcher.start()
        await _warmup()
        runtime.ready = True
        logger.info("server ready")
        yield
        runtime.ready = False
        await runtime.batcher.stop()
        runtime.engine.shutdown()

    async def _warmup():
        """Run one synthetic request through the full pipeline before serving."""
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
        async for _ in synthesize_pcm(runtime, req):
            n += 1
        if n == 0:
            raise RuntimeError("warmup produced no audio")
        logger.info("warmup OK (%d chunks)", n)

    app = FastAPI(lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not runtime.ready:
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
        """Synthesize speech as one response or a chunked audio stream.

        Both response modes use the same incremental LLM and Token2Wav pipeline.
        With ``stream=False``, this handler buffers all PCM before responding;
        it does not switch to the static-batch offline inference path.

        A streaming request resolves its voice before returning response
        headers.  Errors after headers have been sent can only terminate the
        chunked stream.
        """
        media = "audio/wav" if req.response_format == "wav" else "audio/pcm"
        try:
            if req.stream:
                cond_pair = await resolve_condition(runtime, req)
                return StreamingResponse(
                    synthesize_response_chunks(runtime, req, cond_pair),
                    media_type=media)
            pcm_chunks = []
            async with asyncio.timeout(runtime.server_cfg.request_timeout_s):
                async for chunk in synthesize_pcm(runtime, req):
                    pcm_chunks.append(chunk)
            pcm = b"".join(pcm_chunks)
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
            raise HTTPException(504, "request timed out") from e

    @app.post("/v1/audio/voices")
    async def register_voice(req: VoiceRequest):
        def _work():
            wav, sr = decode_ref_audio(req.ref_audio,
                                       runtime.server_cfg.max_ref_seconds)
            return runtime.frontend.process(
                torch.from_numpy(wav), sr, req.ref_text
            )
        try:
            cond = await asyncio.get_running_loop().run_in_executor(None,
                                                                    _work)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        runtime.voices[req.name] = (cond, req.ref_text)
        return {"success": True, "voice": req.name}

    @app.get("/v1/audio/voices")
    async def list_voices():
        return {"voices": sorted(runtime.voices)}

    return app


def main() -> None:
    """Backward-compatible wrapper for the former ``server.app`` entry point."""
    from faster_cosyvoice.server.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
