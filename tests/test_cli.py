from examples.offline_inference import build_argument_parser as offline_parser
from faster_cosyvoice.server.app import build_argument_parser as server_parser


def test_server_help_uses_descriptive_option_names():
    parser = server_parser()
    args = parser.parse_args([])

    assert args.streaming_flow_estimator == "flashinfer"
    assert args.token2wav_batch_mode == "packed"
    assert args.token2wav_batch_size == 8
    assert args.token2wav_scheduler == "deadline"
    assert args.speech_token_chunk_size == 15
    assert args.speech_token_chunk_growth == 2

    help_text = parser.format_help()
    assert "--token2wav-batch-mode" in help_text
    assert "--streaming-flow-graph-buckets" in help_text
    assert "--streaming-vocoder-graph-buckets" in help_text
    assert "--speaker-encoder-tensorrt" in help_text
    assert "--t2w-batch-mode" not in help_text
    assert "--campplus-trt" not in help_text


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
    assert args.streaming_flow_graph_buckets == "512,640"
    assert args.streaming_vocoder_graph_buckets == "64,128"
    assert args.speech_token_chunk_size == 25
    assert args.speech_token_chunk_growth == 1
    assert args.speaker_encoder_tensorrt is True


def test_offline_cli_names_and_legacy_aliases():
    parser = offline_parser()
    args = parser.parse_args([
        "--t2w-cuda-graph-buckets", "8,12",
        "--estimator", "torch",
        "--campplus-trt",
        "--hift-compile",
    ])

    assert args.token2wav_cuda_graph_buckets == "8,12"
    assert args.flow_estimator == "torch"
    assert args.speaker_encoder_tensorrt is True
    assert args.vocoder_compile is True

    help_text = parser.format_help()
    assert "--token2wav-cuda-graph-buckets" in help_text
    assert "--flow-estimator" in help_text
    assert "--speaker-encoder-tensorrt" in help_text
    assert "--t2w-cuda-graph-buckets" not in help_text
