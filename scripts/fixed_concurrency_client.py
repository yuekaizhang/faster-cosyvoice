#!/usr/bin/env python3
"""Run a Nari-compatible TTS benchmark at fixed closed-loop concurrency.

The scheduling model matches CosyVoice's ``client_grpc.py --num-tasks N``:
each worker has at most one request in flight and starts its next request only
after the previous response is complete.  Request transport, PCM validation,
audible-onset detection, playback simulation, metrics, and artifacts are
delegated to the pinned Nari benchmark implementation used by the RPS tests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import aiohttp
from tts_bench import __version__
from tts_bench.artifacts import (
    ArtifactStore,
    initialize_result_directory,
    write_dataset_snapshot,
    write_json,
    write_jsonl,
    write_jsonl_models,
    write_text,
)
from tts_bench.audio import AudibleOnsetDetector
from tts_bench.dataset import PROMPT_ORDER_VERSION, PromptPool, load_prompt_pool
from tts_bench.metrics import derive_summary
from tts_bench.models import (
    ArrivalSlot,
    AudioSuccessCriteria,
    DatasetManifest,
    PcmFormat,
    Phase,
    Prompt,
    RequestRecord,
    RequestSuccessCriteria,
    TargetKind,
    TransportKind,
    WERStatus,
)
from tts_bench.runtime import (
    _benchmark_provenance,
    _execute_request,
    _measurement_records,
    _trace_config,
)
from tts_bench.scheduling import NANOSECONDS_PER_SECOND
from tts_bench.targets import (
    WEBSOCKET_INPUT_CHUNK_INTERVAL_S,
    WEBSOCKET_PROTOCOL,
    TargetAdapter,
    preflight_target,
)

LOAD_MODEL = "closed_loop_fixed_concurrency"
CLIENT_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class FixedConcurrencyOptions:
    concurrency: int
    seed: int
    warmup_s: float
    duration_s: float
    timeout_s: float
    audio_success_criteria: AudioSuccessCriteria = field(default_factory=AudioSuccessCriteria)

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if not math.isfinite(self.warmup_s) or self.warmup_s < 0:
            raise ValueError("warmup must be finite and non-negative")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration must be finite and positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")


@dataclass
class ClosedLoopResult:
    records: list[RequestRecord]
    slots: list[ArrivalSlot]
    peak_in_flight: int
    average_in_flight: float


def parse_duration(value: str) -> float:
    raw = value.strip().lower()
    multiplier = 1.0
    for suffix, factor in (("ms", 0.001), ("s", 1.0), ("m", 60.0)):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            multiplier = factor
            break
    try:
        seconds = float(raw) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid duration: {value!r}") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("duration must be finite and non-negative")
    return seconds


def parse_request_param(value: str) -> tuple[str, Any]:
    key, separator, encoded = value.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("request parameters must use KEY=JSON")
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON request parameter: {error}") from error
    return key.strip(), decoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible TTS endpoint at fixed concurrency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", choices=[kind.value for kind in TargetKind], required=True)
    parser.add_argument(
        "--transport",
        choices=[kind.value for kind in TransportKind],
        default=TransportKind.HTTP.value,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", default="benchmark")
    parser.add_argument("--language", default="English")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup", type=parse_duration, default=15.0)
    parser.add_argument("--duration", type=parse_duration, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=parse_duration, default=120.0)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--request-param", type=parse_request_param, action="append", default=[])
    parser.add_argument(
        "--metadata",
        type=parse_request_param,
        action="append",
        default=[],
        help="Persist extra run provenance as KEY=JSON without adding it to requests.",
    )
    return parser


def measurement_contract(options: FixedConcurrencyOptions) -> dict[str, Any]:
    return {
        "clock": "time.perf_counter_ns",
        "request_start": "first JSON body bytes sent",
        "load_model": LOAD_MODEL,
        "client_protocol_version": CLIENT_PROTOCOL_VERSION,
        "fixed_concurrency": options.concurrency,
        "worker_semantics": (
            "each worker starts its next request only after the complete prior response"
        ),
        "phase_semantics": (
            "warmup and measurement are wall-clock launch windows; active requests finish, and no new "
            "request starts after measurement end"
        ),
        "prompt_assignment": "shared deterministic PromptPool sequence",
        "prompt_order_version": PROMPT_ORDER_VERSION,
        "audible_detector": {
            "version": AudibleOnsetDetector.version,
            "frame_ms": AudibleOnsetDetector.frame_ms,
            "hop_ms": AudibleOnsetDetector.hop_ms,
            "threshold_dbfs": AudibleOnsetDetector.threshold_dbfs,
            "consecutive_active_frames": AudibleOnsetDetector.consecutive_active_frames,
            "dc_removal": "per-frame mean subtraction",
        },
        "playback_startup_buffer_ms": 0,
        "request_success": RequestSuccessCriteria(
            continuity=options.audio_success_criteria
        ).model_dump(mode="json"),
        "wer_evaluator": None,
        "percentile_method": "nearest-rank",
    }


async def run_closed_loop(
    *,
    run_id: str,
    adapter: TargetAdapter,
    pool: PromptPool,
    output: Path,
    store: ArtifactStore,
    options: FixedConcurrencyOptions,
) -> ClosedLoopResult:
    warmup_ns = round(options.warmup_s * NANOSECONDS_PER_SECOND)
    duration_ns = round(options.duration_s * NANOSECONDS_PER_SECOND)
    measurement_end_ns = warmup_ns + duration_ns
    timeout = aiohttp.ClientTimeout(total=options.timeout_s)
    connector = aiohttp.TCPConnector(limit=options.concurrency, force_close=False)
    records: list[RequestRecord] = []
    slots: list[ArrivalSlot] = []
    phase_indexes = {Phase.WARMUP: 0, Phase.MEASUREMENT: 0}
    active_intervals: list[tuple[int, int]] = []
    active_requests = 0
    peak_in_flight = 0
    anchor_ns = perf_counter_ns()

    async def worker(worker_index: int, session: aiohttp.ClientSession) -> None:
        nonlocal active_requests
        nonlocal peak_in_flight
        del worker_index  # Worker identity does not affect deterministic prompt order.
        while True:
            elapsed_ns = max(0, perf_counter_ns() - anchor_ns)
            if elapsed_ns < warmup_ns:
                phase = Phase.WARMUP
            elif elapsed_ns < measurement_end_ns:
                phase = Phase.MEASUREMENT
            else:
                return

            phase_index = phase_indexes[phase]
            phase_indexes[phase] += 1
            prompt = pool.at(phase, phase_index)
            slot = ArrivalSlot(
                phase=phase,
                phase_index=phase_index,
                scheduled_elapsed_ns=elapsed_ns,
                prompt_id=prompt.id,
                prompt_word_count=prompt.word_count,
            )
            slots.append(slot)
            started_ns = max(0, perf_counter_ns() - anchor_ns)
            active_requests += 1
            peak_in_flight = max(peak_in_flight, active_requests)
            try:
                record = await _execute_request(
                    run_id=run_id,
                    adapter=adapter,
                    session=session,
                    prompt=prompt,
                    slot=slot,
                    anchor_ns=anchor_ns,
                    measurement_start_ns=warmup_ns,
                    measurement_end_ns=measurement_end_ns,
                    store=store,
                    audio_success_criteria=options.audio_success_criteria,
                    timeout_s=options.timeout_s,
                )
                records.append(record)
            finally:
                ended_ns = max(0, perf_counter_ns() - anchor_ns)
                active_intervals.append((started_ns, ended_ns))
                active_requests -= 1

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        trace_configs=[_trace_config()],
        headers={"Accept-Encoding": "identity"},
    ) as session:
        tasks = [
            asyncio.create_task(worker(index, session), name=f"closed-loop-worker-{index}")
            for index in range(options.concurrency)
        ]
        await asyncio.gather(*tasks)

    occupied_ns = sum(
        max(0, min(end, measurement_end_ns) - max(start, warmup_ns))
        for start, end in active_intervals
    )
    average_in_flight = occupied_ns / duration_ns
    return ClosedLoopResult(
        records=records,
        slots=slots,
        peak_in_flight=peak_in_flight,
        average_in_flight=average_in_flight,
    )


def custom_report(summary: dict[str, Any]) -> str:
    def metric(name: str) -> str:
        values = summary[name]
        if values["p50"] is None:
            return "n/a"
        return f"{values['p50']:.3f} / {values['p95']:.3f} / {values['p99']:.3f} ms"

    return "\n".join(
        (
            f"Load model: {LOAD_MODEL}",
            f"Fixed concurrency: {summary['fixed_concurrency']}",
            f"Average in flight: {summary['average_in_flight']:.3f}",
            f"Achieved dispatch rate: {summary['actual_rps']:.3f} req/s",
            f"Measurement requests: {summary['started_requests']}",
            f"TTFP p50 / p95 / p99: {metric('first_playable_ms')}",
            f"Audible TTFA p50 / p95 / p99: {metric('audible_ttfa_ms')}",
            f"E2E p50 / p95 / p99: {metric('end_to_end_ms')}",
            (
                "Complete PCM / audible / success: "
                f"{summary['pcm_complete_requests']} / {summary['audible_requests']} / "
                f"{summary['successful_requests']}"
            ),
            (f"Underrun requests: {summary['underrun_requests']} / {summary['started_requests']}"),
            f"Received audio: {summary['received_audio_xrt']:.3f}x realtime",
            "",
        )
    )


async def run_benchmark(
    *,
    adapter: TargetAdapter,
    prompts: tuple[Prompt, ...],
    dataset_manifest: DatasetManifest,
    output: Path,
    options: FixedConcurrencyOptions,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("prompt pool cannot be empty")
    if dataset_manifest.prompt_count != len(prompts):
        raise ValueError("dataset manifest prompt count does not match prompt pool")

    initialize_result_directory(output)
    write_json(output / "status.json", {"schema_version": 1, "state": "initializing"})
    write_dataset_snapshot(output, prompts, dataset_manifest)
    run_id = uuid4().hex
    store = ArtifactStore(output)
    store.start()
    try:
        setup = await preflight_target(adapter, timeout_s=options.timeout_s)
        pool = PromptPool(prompts, seed=options.seed)
        benchmark_revision, benchmark_dirty, dependency_lock_sha256 = _benchmark_provenance()
        run_configuration = {
            "schema_version": 1,
            "run_id": run_id,
            "package_version": __version__,
            "benchmark_git_revision": benchmark_revision,
            "benchmark_git_dirty": benchmark_dirty,
            "dependency_lock_sha256": dependency_lock_sha256,
            "target": adapter.kind.value,
            "transport": adapter.transport.value,
            "base_url": adapter.base_url,
            "model": adapter.model,
            "voice": adapter.voice,
            "language": adapter.language,
            "load_model": LOAD_MODEL,
            "fixed_concurrency": options.concurrency,
            "seed": options.seed,
            "warmup_s": options.warmup_s,
            "duration_s": options.duration_s,
            "timeout_s": options.timeout_s,
            "sample_rate_hz": adapter.expected_format.sample_rate_hz,
            "health_path": adapter.health_path,
            "request_params": adapter.request_params,
            "request_body_template": adapter.request_template(),
            "request_body_example": adapter.build_request(pool.at(Phase.MEASUREMENT, 0).text),
            "request_records_path": "requests.jsonl",
            "measurement_contract": measurement_contract(options),
            "dataset": dataset_manifest.model_dump(mode="json"),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "client": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
                "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "setup": setup.model_dump(mode="json"),
            "metadata": metadata or {},
        }
        if adapter.transport is TransportKind.WEBSOCKET:
            run_configuration["measurement_contract"]["request_start"] = (
                "first input_text.append frame sent"
            )
            run_configuration["measurement_contract"]["websocket_input"] = {
                "protocol": WEBSOCKET_PROTOCOL,
                "chunk_unit": "word",
                "chunk_size": 1,
                "chunk_interval_ms": WEBSOCKET_INPUT_CHUNK_INTERVAL_S * 1000,
                "acknowledgement": "required before the next chunk",
                "cadence": "absolute from the first text chunk",
            }
        write_json(output / "run.json", run_configuration)
        write_json(output / "status.json", {"schema_version": 1, "state": "running"})

        result = await run_closed_loop(
            run_id=run_id,
            adapter=adapter,
            pool=pool,
            output=output,
            store=store,
            options=options,
        )
        await store.close()
        ordered_slots = sorted(
            result.slots,
            key=lambda slot: (0 if slot.phase is Phase.WARMUP else 1, slot.phase_index),
        )
        write_jsonl_models(output / "arrivals.jsonl", ordered_slots)
        records = [
            record.model_copy(
                update={"audio_success": record.success, "wer_status": WERStatus.DISABLED}
            )
            for record in result.records
        ]
        write_jsonl_models(output / "requests.jsonl", records)
        measurement_records = _measurement_records(records)
        write_jsonl(
            output / "measurement-prompt-sequence.jsonl",
            (
                {
                    "phase_index": record.phase_index,
                    "request_id": record.request_id,
                    "prompt_id": record.prompt_id,
                    "prompt_word_count": record.prompt_word_count,
                    "dispatch_elapsed_ns": record.dispatch_elapsed_ns,
                }
                for record in measurement_records
            ),
        )
        warmup_ns = round(options.warmup_s * NANOSECONDS_PER_SECOND)
        duration_ns = round(options.duration_s * NANOSECONDS_PER_SECOND)
        nari_summary = derive_summary(
            run_id=run_id,
            target=adapter.kind,
            transport=adapter.transport,
            requested_rps=1.0,  # Compatibility-only input; removed from persisted summary below.
            warmup_ns=warmup_ns,
            duration_ns=duration_ns,
            scheduled_requests=len(measurement_records),
            dropped_requests=0,
            records=records,
            peak_in_flight=result.peak_in_flight,
            setup=setup,
            wer_enabled=False,
        )
        summary = nari_summary.model_dump(mode="json")
        summary.pop("requested_rps", None)
        summary.update(
            {
                "load_model": LOAD_MODEL,
                "client_protocol_version": CLIENT_PROTOCOL_VERSION,
                "fixed_concurrency": options.concurrency,
                "average_in_flight": result.average_in_flight,
            }
        )
        write_json(output / "summary.json", summary)
        write_text(output / "report.txt", custom_report(summary))
        write_json(
            output / "status.json",
            {
                "schema_version": 1,
                "state": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return summary
    except BaseException as error:
        with contextlib.suppress(BaseException):
            await store.close()
        write_json(
            output / "status.json",
            {
                "schema_version": 1,
                "state": "incomplete",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def main() -> None:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    request_params = dict(args.request_param)
    prompts, manifest = load_prompt_pool(args.dataset.expanduser().resolve())
    adapter = TargetAdapter(
        kind=TargetKind(args.target),
        base_url=args.base_url,
        model=args.model,
        voice=args.voice,
        language=args.language,
        expected_format=PcmFormat(sample_rate_hz=24_000),
        request_params=request_params,
        health_path=args.health_path,
        transport=TransportKind(args.transport),
    )
    options = FixedConcurrencyOptions(
        concurrency=args.concurrency,
        seed=args.seed,
        warmup_s=args.warmup,
        duration_s=args.duration,
        timeout_s=args.timeout,
    )
    summary = asyncio.run(
        run_benchmark(
            adapter=adapter,
            prompts=prompts,
            dataset_manifest=manifest,
            output=args.output.expanduser().resolve(),
            options=options,
            metadata=dict(args.metadata),
        )
    )
    print(custom_report(summary), end="")


if __name__ == "__main__":
    main()
