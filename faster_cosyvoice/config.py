# faster_cosyvoice/config.py
"""全部旋钮。默认值 = spec 附录 A。"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    target_model: str = "yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF"
    draft_model: Optional[str] = "yuekai/cosyvoice3_llm_dspark"  # None = 关投机解码
    method: Optional[str] = None            # None → 从 draft config.json 自动
    num_spec_tokens: Optional[int] = None   # None → block_size - 1
    draft_sample_method: str = "probabilistic"
    # offline 默认 0.6：dspark draft 的 KV/graphs (~12GB) 不计入 vLLM 配额，
    # 且 token2wav/frontend 同卡常驻；0.8 会 OOM (80GB H100)。server(M2) 用 0.5
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 4096
    # CV3 官方采样参数（spec 附录 A；rp=1.0 会跑飞）
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 15
    repetition_penalty: float = 1.1
    max_tokens: int = 2048  # 上限；server(M2) 实际 min(2048, 20×text_len)


@dataclass
class Token2WavConfig:
    model_dir: str = "models/Fun-CosyVoice3-0.5B-2512"
    device: str = "cuda:0"
    estimator_mode: str = "flashinfer"      # flashinfer | torch
    batch_size: int = 8                     # flashinfer packed 子批上限（防 OOM）
    batch_mode: str = "packed"              # [M3] packed | serial（batcher v2；与 CLI 默认一致）
    # 逗号分隔秒数字符串（总时长 prompt+generated，如 "8,12,16,20,24"）；
    # None=关。仅 offline batch=1（CFG 双行 b==2）走 bucketed CUDA graph。
    cuda_graph_buckets: Optional[str] = None


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.5   # server 档（spec D6；与 token2wav 同卡）
    max_ref_seconds: float = 30.0         # ref 音频超长截断并告警（spec §7）
    request_timeout_s: float = 300.0
    voice_cache_size: int = 256
