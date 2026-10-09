"""Server command-line arguments and conversion to runtime settings."""

import argparse

from faster_cosyvoice.config import (
    DEFAULT_STREAMING_FLOW_GRAPH_BUCKETS,
    DEFAULT_STREAMING_VOCODER_GRAPH_BUCKETS,
    LLMConfig,
    ServerConfig,
    Token2WavConfig,
    graph_buckets,
)


def _hidden_alias(container, *names: str, **kwargs) -> None:
    """Add a backward-compatible option without showing it in ``--help``."""
    container.add_argument(
        *names,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
        **kwargs,
    )


def _add_server_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Server")
    group.add_argument("--host", default="0.0.0.0")
    group.add_argument("--port", type=int, default=8000)
    group.add_argument(
        "--request-timeout-s",
        type=float,
        default=300.0,
        help="Timeout for buffered responses; streaming timeouts are controlled "
        "by the client (default: 300).",
    )


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Speech-token LLM")
    group.add_argument(
        "--target-model", default="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF"
    )
    group.add_argument("--draft-model", default="yuekai/cosyvoice3_llm_dspark")
    group.add_argument("--gpu-memory-utilization", type=float, default=0.5)


def _add_token2wav_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Token2Wav")
    group.add_argument(
        "--token2wav-dir", default="models/Fun-CosyVoice3-0.5B-2512"
    )
    group.add_argument("--token2wav-device", default="cuda:0")
    group.add_argument(
        "--streaming-flow-estimator",
        default="flashinfer",
        choices=["torch", "flashinfer"],
        help="Flow estimator used for streaming synthesis (default: "
        "flashinfer; torch is the fallback).",
    )
    _hidden_alias(
        group,
        "--stream-estimator",
        dest="streaming_flow_estimator",
        choices=["torch", "flashinfer"],
    )
    group.add_argument(
        "--token2wav-batch-mode",
        default="packed",
        choices=["serial", "packed"],
        help="Cross-session Token2Wav execution mode (default: packed; "
        "requires FlashInfer).",
    )
    _hidden_alias(
        group,
        "--t2w-batch-mode",
        dest="token2wav_batch_mode",
        choices=["serial", "packed"],
    )
    group.add_argument(
        "--token2wav-batch-size",
        type=int,
        default=8,
        help="Maximum packed Token2Wav batch size (default: 8).",
    )
    _hidden_alias(
        group,
        "--t2w-batch-size",
        dest="token2wav_batch_size",
        type=int,
    )
    group.add_argument(
        "--token2wav-deadline-reserve-ms",
        type=float,
        default=100.0,
        help="Prioritize an established stream when its playback buffer is "
        "this close to empty (default: 100 ms).",
    )
    _hidden_alias(
        group,
        "--t2w-deadline-reserve-ms",
        dest="token2wav_deadline_reserve_ms",
        type=float,
    )
    group.add_argument(
        "--token2wav-scheduler",
        default="deadline",
        choices=["legacy", "deadline"],
        help="Token2Wav scheduling policy (default: deadline; legacy is "
        "retained for compatibility experiments).",
    )
    _hidden_alias(
        group,
        "--t2w-scheduler",
        dest="token2wav_scheduler",
        choices=["legacy", "deadline"],
    )
    group.add_argument(
        "--vocoder-compile",
        action="store_true",
        help="Compile the HiFT vocoder and quantize offline input lengths to "
        "64-Mel-frame buckets. Adds 15-20 seconds of one-time startup warmup.",
    )
    _hidden_alias(
        group,
        "--hift-compile",
        dest="vocoder_compile",
        action="store_true",
    )
    group.add_argument(
        "--speaker-encoder-tensorrt",
        action="store_true",
        help="Run the CampPlus speaker encoder with TensorRT instead of ONNX "
        "Runtime CPU. The first run builds and caches an engine.",
    )
    _hidden_alias(
        group,
        "--campplus-trt",
        dest="speaker_encoder_tensorrt",
        action="store_true",
    )


def _add_cuda_graph_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Streaming CUDA Graph")
    group.add_argument(
        "--streaming-cuda-graph",
        action="store_true",
        help="Enable streaming Flow and HiFT CUDA Graphs with built-in bucket "
        "presets. Use the per-module bucket options only for advanced tuning.",
    )

    flow = group.add_mutually_exclusive_group()
    flow.add_argument(
        "--streaming-flow-graph-bucket-seconds",
        dest="streaming_flow_graph_buckets",
        type=graph_buckets.parse_mel_seconds,
        metavar="SECONDS",
        default=None,
        help="Comma-separated full-context duration buckets in seconds for "
        "single-session streaming Flow CUDA Graph, for example "
        "10.24,12.8,15.36,17.92,20.48,25.6.",
    )
    _hidden_alias(
        flow,
        "--streaming-flow-graph-buckets",
        "--stream-graph-buckets",
        dest="streaming_flow_graph_buckets",
        type=graph_buckets.parse_mel_frames,
    )

    vocoder = group.add_mutually_exclusive_group()
    vocoder.add_argument(
        "--streaming-vocoder-graph-bucket-seconds",
        dest="streaming_vocoder_graph_buckets",
        type=graph_buckets.parse_mel_seconds,
        metavar="SECONDS",
        default=None,
        help="Comma-separated input-audio duration buckets in seconds for "
        "streaming HiFT CUDA Graph, for example "
        "1.28,2.56,3.84,5.12,7.68,10.24. Non-final chunks only.",
    )
    _hidden_alias(
        vocoder,
        "--streaming-vocoder-graph-buckets",
        "--hift-graph-buckets",
        dest="streaming_vocoder_graph_buckets",
        type=graph_buckets.parse_mel_frames,
    )


def _add_audio_output_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Streaming audio")
    group.add_argument(
        "--speech-token-chunk-size",
        type=int,
        default=15,
        help="Speech tokens consumed by the first streaming hop (default: 15; "
        "use 25 for fixed-shape graph mode).",
    )
    _hidden_alias(
        group,
        "--codec-chunk-frames",
        dest="speech_token_chunk_size",
        type=int,
    )
    group.add_argument(
        "--speech-token-chunk-growth",
        type=int,
        default=2,
        help="Multiplier applied to successive streaming hops (default: 2; "
        "use 1 for fixed-size hops).",
    )
    _hidden_alias(
        group,
        "--codec-chunk-scale",
        dest="speech_token_chunk_growth",
        type=int,
    )
    group.add_argument(
        "--trim-leading-silence",
        action="store_true",
        help="Trim bounded leading silence while retaining pre-roll. This "
        "changes only the beginning of the PCM stream.",
    )
    group.add_argument(
        "--leading-silence-preroll-ms",
        type=float,
        default=20.0,
        help="Audio retained before detected speech (default: 20 ms).",
    )
    group.add_argument(
        "--leading-silence-max-ms",
        type=float,
        default=2000.0,
        help="Maximum leading-silence window (default: 2000 ms).",
    )
    group.add_argument(
        "--leading-silence-min-buffer-ms",
        type=float,
        default=400.0,
        help="Playable audio buffered before the first trimmed response chunk "
        "(default: 400 ms).",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the public server CLI with options grouped by subsystem."""
    parser = argparse.ArgumentParser(
        description="Serve CosyVoice3 through an OpenAI-compatible speech API."
    )
    _add_server_arguments(parser)
    _add_llm_arguments(parser)
    _add_token2wav_arguments(parser)
    _add_cuda_graph_arguments(parser)
    _add_audio_output_arguments(parser)
    return parser


def resolve_streaming_graph_buckets(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Apply built-in graph presets without overriding explicit bucket tuning."""
    flow_buckets = args.streaming_flow_graph_buckets
    vocoder_buckets = args.streaming_vocoder_graph_buckets
    if args.streaming_cuda_graph:
        if flow_buckets is None:
            flow_buckets = DEFAULT_STREAMING_FLOW_GRAPH_BUCKETS
        if vocoder_buckets is None:
            vocoder_buckets = DEFAULT_STREAMING_VOCODER_GRAPH_BUCKETS
    return flow_buckets, vocoder_buckets


def validate_server_args(args: argparse.Namespace) -> None:
    """Reject combinations that cannot be represented by runtime settings."""
    if (
        args.token2wav_batch_mode == "packed"
        and args.streaming_flow_estimator != "flashinfer"
    ):
        raise SystemExit(
            "--token2wav-batch-mode packed requires "
            "--streaming-flow-estimator flashinfer; use "
            "--token2wav-batch-mode serial with the torch estimator"
        )
    if args.token2wav_batch_size < 1:
        raise SystemExit("--token2wav-batch-size must be >= 1")
    if args.token2wav_deadline_reserve_ms < 0:
        raise SystemExit("--token2wav-deadline-reserve-ms must be >= 0")
    if (
        args.leading_silence_preroll_ms < 0
        or args.leading_silence_max_ms < 0
        or args.leading_silence_min_buffer_ms < 0
        or args.leading_silence_preroll_ms > args.leading_silence_max_ms
    ):
        raise SystemExit(
            "require 0 <= --leading-silence-preroll-ms "
            "<= --leading-silence-max-ms"
        )


def build_server_configs(
    args: argparse.Namespace,
) -> tuple[LLMConfig, Token2WavConfig, ServerConfig]:
    """Translate parsed CLI arguments into the three runtime settings objects."""
    draft_model = None if args.draft_model in (None, "none") else args.draft_model
    flow_graph_buckets, vocoder_graph_buckets = resolve_streaming_graph_buckets(args)
    llm_config = LLMConfig(
        target_model=args.target_model,
        draft_model=draft_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    token2wav_config = Token2WavConfig(
        model_dir=args.token2wav_dir,
        device=args.token2wav_device,
        estimator_mode=args.streaming_flow_estimator,
        batch_mode=args.token2wav_batch_mode,
        batch_size=args.token2wav_batch_size,
        scheduler_mode=args.token2wav_scheduler,
        deadline_reserve_s=args.token2wav_deadline_reserve_ms / 1000,
        vocoder_compile=args.vocoder_compile,
        speaker_encoder_tensorrt=args.speaker_encoder_tensorrt,
        streaming_flow_graph_buckets=flow_graph_buckets,
        streaming_vocoder_graph_buckets=vocoder_graph_buckets,
    )
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        request_timeout_s=args.request_timeout_s,
        speech_token_chunk_size=args.speech_token_chunk_size,
        speech_token_chunk_growth=args.speech_token_chunk_growth,
        trim_leading_silence=args.trim_leading_silence,
        leading_silence_preroll_ms=args.leading_silence_preroll_ms,
        leading_silence_max_ms=args.leading_silence_max_ms,
        leading_silence_min_buffer_ms=args.leading_silence_min_buffer_ms,
    )
    return llm_config, token2wav_config, server_config
