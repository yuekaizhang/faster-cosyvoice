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
    # opt-in torch.compile(hift.decode) + mel 长度 pad-to-bucket（64 帧粒度）：
    # eager hift 每遇新 mel 长度要付一次 cudnn v8 plan-build（fresh-shape
    # ~52-55ms/次，warm 同长度 ~19ms；每条请求 mel 长度都不同 → fresh 是常态）。
    # 单纯 compile 不解决：inductor 的 conv1d 仍落回 ATen/cudnn（triton conv
    # 模板对本形状无 choice），fresh 仍 ~49ms；配合 pad-to-bucket 把长度空间收
    # 敛到少数桶（init warmup 预热到 1280 帧），全部命中 warm plan →
    # hift 整段 ~13-21ms（~3x）。一次性 warmup（编译+桶预热）~15-20s，init 付清。
    # 波形 vs eager 数值差：offline ~8e-4；流式逐 chunk 放大，worst-chunk
    # ~6e-2（inductor 融合 + pad 改变 cudnn 算法选择；质量门以 ASR CER 为准，
    # 若需要流式逐位稳定请保持关闭），默认关。
    hift_compile: bool = False
    # opt-in campplus speaker embedding 走 TensorRT（默认 ORT-CPU）：冷 ref
    # resolve 88.6→23.3ms（spk_emb ~58→~7ms）。首启一次性 build ~2-3min 存
    # campplus.<gpu>.fp32.plan；embedding 数值差 ~1e-5（ASR CER 门通过）。
    campplus_trt: bool = False


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.5   # server 档（spec D6；与 token2wav 同卡）
    max_ref_seconds: float = 30.0         # ref 音频超长截断并告警（spec §7）
    request_timeout_s: float = 300.0
    voice_cache_size: int = 256
