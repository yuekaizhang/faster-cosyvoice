import pytest

from examples.offline_inference import (
    build_argument_parser as offline_parser,
)
from examples.offline_inference import (
    resolve_offline_flow_graph_buckets,
)
from faster_cosyvoice.config import (
    DEFAULT_OFFLINE_FLOW_GRAPH_BUCKET_SECONDS,
    DEFAULT_STREAMING_FLOW_GRAPH_BUCKETS,
    DEFAULT_STREAMING_VOCODER_GRAPH_BUCKETS,
)
from faster_cosyvoice.server.arguments import (
    build_argument_parser as server_parser,
)
from faster_cosyvoice.server.arguments import (
    resolve_streaming_graph_buckets,
)


def test_server_help_uses_descriptive_option_names():
    parser = server_parser()
    args = parser.parse_args([])

    assert args.streaming_flow_estimator == "flashinfer"
    assert args.token2wav_batch_mode == "packed"
    assert args.token2wav_batch_size == 8
    assert args.token2wav_scheduler == "deadline"
    assert args.speech_token_chunk_size == 15
    assert args.speech_token_chunk_growth == 2
    assert args.streaming_cuda_graph is False
    assert args.streaming_flow_graph_buckets is None
    assert args.streaming_vocoder_graph_buckets is None

    help_text = parser.format_help()
    assert "--token2wav-batch-mode" in help_text
    assert "--streaming-cuda-graph" in help_text
    assert "--streaming-flow-graph-bucket-seconds" in help_text
    assert "--streaming-vocoder-graph-bucket-seconds" in help_text
    assert "--speaker-encoder-tensorrt" in help_text
    assert "--t2w-batch-mode" not in help_text
    assert "--streaming-flow-graph-buckets" not in help_text
    assert "--streaming-vocoder-graph-buckets" not in help_text
    assert "--campplus-trt" not in help_text


def test_server_cuda_graph_switch_uses_built_in_buckets():
    args = server_parser().parse_args(["--streaming-cuda-graph"])

    assert resolve_streaming_graph_buckets(args) == (
        DEFAULT_STREAMING_FLOW_GRAPH_BUCKETS,
        DEFAULT_STREAMING_VOCODER_GRAPH_BUCKETS,
    )


def test_server_cuda_graph_switch_preserves_explicit_bucket_override():
    args = server_parser().parse_args([
        "--streaming-cuda-graph",
        "--streaming-flow-graph-bucket-seconds", "10.24,12.8",
    ])

    assert resolve_streaming_graph_buckets(args) == (
        (512, 640),
        DEFAULT_STREAMING_VOCODER_GRAPH_BUCKETS,
    )


def test_server_graph_bucket_seconds_convert_to_mel_frames():
    args = server_parser().parse_args([
        "--streaming-flow-graph-bucket-seconds", "10.24,12.8",
        "--streaming-vocoder-graph-bucket-seconds", "1.28,2.56",
    ])

    assert args.streaming_flow_graph_buckets == (512, 640)
    assert args.streaming_vocoder_graph_buckets == (64, 128)


def test_server_graph_bucket_seconds_require_20_ms_alignment():
    with pytest.raises(SystemExit):
        server_parser().parse_args([
            "--streaming-flow-graph-bucket-seconds", "1.29",
        ])


def test_server_legacy_options_are_hidden_aliases():
    args = server_parser().parse_args([
        "--stream-estimator", "torch",
        "--t2w-batch-mode", "serial",
        "--t2w-batch-size", "3",
        "--t2w-deadline-reserve-ms", "40",
        "--t2w-scheduler", "legacy",
        "--hift-compile",
        "--stream-graph-buckets", "512,640",
        "--hift-graph-buckets", "64,128",
        "--codec-chunk-frames", "25",
        "--codec-chunk-scale", "1",
        "--campplus-trt",
    ])

    assert args.streaming_flow_estimator == "torch"
    assert args.token2wav_batch_mode == "serial"
    assert args.token2wav_batch_size == 3
    assert args.token2wav_deadline_reserve_ms == 40
    assert args.token2wav_scheduler == "legacy"
    assert args.vocoder_compile is True
    assert args.streaming_flow_graph_buckets == (512, 640)
    assert args.streaming_vocoder_graph_buckets == (64, 128)
    assert args.speech_token_chunk_size == 25
    assert args.speech_token_chunk_growth == 1
    assert args.speaker_encoder_tensorrt is True


def test_offline_cli_names_and_legacy_aliases():
    parser = offline_parser()
    args = parser.parse_args([
        "--offline-flow-graph-bucket-seconds", "8,12",
        "--estimator", "torch",
        "--campplus-trt",
        "--hift-compile",
    ])

    assert args.offline_flow_graph_duration_buckets == (8.0, 12.0)
    assert args.flow_estimator == "torch"
    assert args.speaker_encoder_tensorrt is True
    assert args.vocoder_compile is True

    help_text = parser.format_help()
    assert "--offline-flow-cuda-graph" in help_text
    assert "--offline-flow-graph-bucket-seconds" in help_text
    assert "--flow-estimator" in help_text
    assert "--speaker-encoder-tensorrt" in help_text
    assert "--token2wav-cuda-graph-buckets" not in help_text
    assert "--t2w-cuda-graph-buckets" not in help_text

    legacy = parser.parse_args(["--t2w-cuda-graph-buckets", "8,12"])
    assert legacy.offline_flow_graph_duration_buckets == (8.0, 12.0)


def test_offline_cuda_graph_switch_uses_built_in_buckets():
    args = offline_parser().parse_args(["--offline-flow-cuda-graph"])
    assert (
        resolve_offline_flow_graph_buckets(args)
        == DEFAULT_OFFLINE_FLOW_GRAPH_BUCKET_SECONDS
    )


def test_offline_cuda_graph_switch_preserves_explicit_bucket_override():
    args = offline_parser().parse_args([
        "--offline-flow-cuda-graph",
        "--offline-flow-graph-bucket-seconds", "10,14",
    ])
    assert resolve_offline_flow_graph_buckets(args) == (10.0, 14.0)
