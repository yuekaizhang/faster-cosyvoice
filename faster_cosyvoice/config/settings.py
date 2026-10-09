"""Typed runtime settings and their documented defaults."""

from dataclasses import dataclass
from typing import Optional

# CUDA Graph presets used by the simple enable switches.  The CLI exposes
# durations in seconds for manual tuning, while the streaming runtime stores
# its buckets in 50 Hz Mel frames.
DEFAULT_OFFLINE_FLOW_GRAPH_BUCKET_SECONDS = (8.0, 12.0, 16.0, 20.0, 24.0)
DEFAULT_STREAMING_FLOW_GRAPH_BUCKETS = (512, 640, 768, 896, 1024, 1280)
DEFAULT_STREAMING_VOCODER_GRAPH_BUCKETS = (64, 128, 192, 256, 384, 512)


@dataclass
class LLMConfig:
    target_model: str = "yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF"
    draft_model: Optional[str] = "yuekai/cosyvoice3_llm_dspark"
    method: Optional[str] = None
    num_spec_tokens: Optional[int] = None
    draft_sample_method: str = "probabilistic"
    # DSpark's KV cache and graphs are outside vLLM's memory budget.  Keep
    # enough memory available for the token2wav models on a shared GPU.
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 4096
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 15
    repetition_penalty: float = 1.1
    max_tokens: int = 2048


@dataclass
class Token2WavConfig:
    model_dir: str = "models/Fun-CosyVoice3-0.5B-2512"
    device: str = "cuda:0"
    estimator_mode: str = "flashinfer"      # flashinfer | torch
    batch_size: int = 8                     # maximum packed sub-batch
    batch_mode: str = "packed"              # packed | serial
    scheduler_mode: str = "deadline"        # deadline | legacy
    deadline_reserve_s: float = 0.1

    # Offline Flow graph buckets are total prompt + generated audio duration
    # in seconds.  Only token2wav batches of one use this graph path.
    offline_flow_graph_duration_buckets: Optional[tuple[float, ...]] = None

    # Streaming Flow graph buckets are full sequence lengths in Mel frames.
    # The sequence contains prompt, first-chunk padding, and consumed tokens.
    streaming_flow_graph_buckets: Optional[tuple[int, ...]] = None

    # torch.compile the HiFT vocoder and round offline Mel lengths up to the
    # implementation's 64-frame buckets.
    vocoder_compile: bool = False

    # Streaming HiFT CUDA Graph input lengths in Mel frames.  They apply only
    # to non-final chunks; final and overlong chunks use the eager path.
    streaming_vocoder_graph_buckets: Optional[tuple[int, ...]] = None

    # Use TensorRT for the CampPlus speaker encoder instead of ONNX Runtime CPU.
    speaker_encoder_tensorrt: bool = False


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.5
    max_ref_seconds: float = 30.0
    request_timeout_s: float = 300.0
    voice_cache_size: int = 256

    # Fixed-shape graph mode uses size=25 and growth=1.  The defaults retain
    # the original 15-token hop with 2x growth.
    speech_token_chunk_size: int = 15
    speech_token_chunk_growth: int = 2

    trim_leading_silence: bool = False
    leading_silence_preroll_ms: float = 20.0
    leading_silence_max_ms: float = 2000.0
    leading_silence_min_buffer_ms: float = 400.0
