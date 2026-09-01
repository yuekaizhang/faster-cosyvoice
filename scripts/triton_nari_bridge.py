#!/usr/bin/env python3
"""Expose the CosyVoice Triton gRPC stream as Nari's PCM HTTP endpoint.

The bridge deliberately does no request scheduling or audio buffering: every
Triton waveform response is converted to little-endian PCM16 and written to the
HTTP response immediately.  This lets ``tts-bench --target nari`` measure the
same first generated audio chunk that the native Triton client observes.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import signal
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpcclient
from aiohttp import web
from scipy.signal import resample_poly
from tritonclient.utils import np_to_triton_dtype

LOG = logging.getLogger("triton-nari-bridge")


def _decode_audio(source: str | Path | bytes, sample_rate: int = 16_000) -> np.ndarray:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    waveform, source_rate = sf.read(source, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if source_rate != sample_rate:
        divisor = np.gcd(source_rate, sample_rate)
        waveform = resample_poly(
            waveform, sample_rate // divisor, source_rate // divisor
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _load_arrow_reference(
    arrow_path: Path,
    row_index: int,
    audio_column: str,
    text_column: str,
) -> tuple[np.ndarray, str]:
    # Read Arrow directly so this also works in NVIDIA Triton images whose
    # bundled `datasets` version may not understand a newer cached schema.
    import pyarrow as pa

    with pa.memory_map(str(arrow_path), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    row = table.slice(row_index, 1).to_pylist()[0]
    audio = row[audio_column]
    if audio.get("bytes") is not None:
        waveform = _decode_audio(audio["bytes"])
    else:
        audio_path = Path(audio["path"])
        if not audio_path.is_absolute():
            audio_path = arrow_path.parent / audio_path
        waveform = _decode_audio(audio_path)
    return waveform, row[text_column]


def load_reference(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.reference_arrow:
        return _load_arrow_reference(
            args.reference_arrow,
            args.reference_row,
            args.reference_audio_column,
            args.reference_text_column,
        )
    if not args.reference_wav or not args.reference_text:
        raise ValueError(
            "provide --reference-arrow, or both --reference-wav and --reference-text"
        )
    return _decode_audio(args.reference_wav), args.reference_text


def _make_inputs(
    waveform: np.ndarray, reference_text: str, target_text: str
) -> tuple[list[grpcclient.InferInput], list[grpcclient.InferRequestedOutput]]:
    samples = waveform.reshape(1, -1)
    lengths = np.array([[samples.shape[1]]], dtype=np.int32)
    reference = np.array([[reference_text]], dtype=object)
    target = np.array([[target_text]], dtype=object)
    values = (
        ("reference_wav", samples),
        ("reference_wav_len", lengths),
        ("reference_text", reference),
        ("target_text", target),
    )
    inputs = []
    for name, value in values:
        datatype = "BYTES" if value.dtype == object else np_to_triton_dtype(value.dtype)
        item = grpcclient.InferInput(name, value.shape, datatype)
        item.set_data_from_numpy(value)
        inputs.append(item)
    return inputs, [grpcclient.InferRequestedOutput("waveform")]


def _pcm16(waveform: np.ndarray) -> bytes:
    # Match soundfile's conventional PCM16 quantization while making the wire
    # format explicit for tts-bench (mono, 24 kHz, signed little endian).
    clipped = np.clip(waveform.reshape(-1), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2", copy=False).tobytes()


class Bridge:
    def __init__(
        self,
        triton_url: str,
        model_name: str,
        reference_waveform: np.ndarray,
        reference_text: str,
        timeout_s: float,
    ) -> None:
        self.client = grpcclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.reference_waveform = reference_waveform
        self.reference_text = reference_text
        self.timeout_s = timeout_s

    async def close(self) -> None:
        await self.client.close()

    async def healthy(self) -> bool:
        try:
            return bool(
                await self.client.is_server_live()
                and await self.client.is_server_ready()
                and await self.client.is_model_ready(self.model_name)
            )
        except Exception:
            return False

    async def health(self, _request: web.Request) -> web.Response:
        if await self.healthy():
            return web.json_response({"status": "ok"})
        return web.json_response({"status": "unavailable"}, status=503)

    async def speech(self, request: web.Request) -> web.StreamResponse:
        try:
            payload = await request.json()
        except Exception as error:
            raise web.HTTPBadRequest(text=f"invalid JSON: {error}") from error
        target_text = payload.get("input")
        if not isinstance(target_text, str) or not target_text.strip():
            raise web.HTTPBadRequest(text="'input' must be a non-empty string")

        inputs, outputs = _make_inputs(
            self.reference_waveform, self.reference_text, target_text
        )
        request_id = uuid.uuid4().hex

        async def requests():
            yield {
                "model_name": self.model_name,
                "inputs": inputs,
                "outputs": outputs,
                "request_id": request_id,
                "parameters": {"triton_enable_empty_final_response": True},
            }

        results = self.client.stream_infer(requests(), stream_timeout=self.timeout_s)
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "audio/pcm",
                "Cache-Control": "no-store",
                "X-Audio-Sample-Rate": "24000",
            },
        )
        await response.prepare(request)
        try:
            async for result, error in results:
                if error is not None:
                    raise RuntimeError(str(error))
                parameters = result.get_response().parameters
                if (
                    "triton_final_response" in parameters
                    and parameters["triton_final_response"].bool_param
                ):
                    break
                waveform = result.as_numpy("waveform")
                if waveform is not None and waveform.size:
                    await response.write(_pcm16(waveform))
            await response.write_eof()
        except (ConnectionError, asyncio.CancelledError):
            results.cancel()
            raise
        except Exception:
            results.cancel()
            LOG.exception("Triton request %s failed", request_id)
            # Headers may already be on the wire, so terminate this response;
            # tts-bench will record the truncated request rather than fake audio.
            response.force_close()
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--triton-url", default="127.0.0.1:18001")
    parser.add_argument("--model-name", default="cosyvoice3")
    parser.add_argument("--timeout", type=float, default=120.0)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-arrow", type=Path)
    source.add_argument("--reference-wav", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--reference-row", type=int, default=0)
    parser.add_argument("--reference-audio-column", default="prompt_audio")
    parser.add_argument("--reference-text-column", default="prompt_text")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    waveform, reference_text = load_reference(args)
    bridge = Bridge(
        args.triton_url,
        args.model_name,
        waveform,
        reference_text,
        args.timeout,
    )
    app = web.Application()
    app.router.add_get("/health", bridge.health)
    app.router.add_post("/v1/audio/speech", bridge.speech)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    LOG.info(
        "listening on http://%s:%d; Triton=%s model=%s reference_samples=%d",
        args.host,
        args.port,
        args.triton_url,
        args.model_name,
        waveform.size,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await runner.cleanup()
    await bridge.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
