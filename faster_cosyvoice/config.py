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
    scheduler_mode: str = "deadline"        # deadline | legacy（仅供严格消融）
    # Nari-style playback-deadline reserve：既有 stream 剩余 buffer 小于该值
    # 时抢占 startup work。当前全前缀重算路径保守留 100ms；可由 CLI 调优。
    deadline_reserve_s: float = 0.1
    # 逗号分隔秒数字符串（总时长 prompt+generated，如 "8,12,16,20,24"）；
    # None=关。仅 offline batch=1（CFG 双行 b==2）走 bucketed CUDA graph。
    cuda_graph_buckets: Optional[str] = None
    # [M3.5-r2] 流式 bucketed CUDA graphs：逗号分隔 mel 帧数（50fps，如
    # "512,640,768,896,1024,1280"）；None=关。仅单 session 流式（CFG 双行
    # b==2、streaming=True）命中：graph 内 dense-SDPA + 运行时
    # chunk-causal&true-len 掩码，与 eager flashinfer ragged 非逐位一致
    # （opt-in；质量门 = ASR CER）。chunk-k flow 序列 =
    # (prompt+pad+consumed)*2 帧，逐 voice 确定：首 chunk（uniform-25 下
    # ≈(prompt+pad+25)*2）附近配一个细 bucket 保 TTFP。每 bucket 首遇 lazy
    # capture ~100ms（server warmup 会预热 warmup voice 命中的桶；真实 voice
    # prompt 长度不同 → 每桶首个真实请求付一次 capture）。
    stream_graph_buckets: Optional[str] = None
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
    # [M3.5-r4] 流式 hift bucketed CUDA graphs：逗号分隔 mel 帧数（如
    # "64,128,192,256,384,512"）；None=关。finalize=False 中间 chunk 的
    # hift.inference 整段捕成 per-bucket graph（pad-to-bucket + 按真实长度
    # 切片，数学等价，实测 max-abs-diff ~3e-4；finalize=True 最终 chunk 走原
    # 路径）。同卡 vLLM 抢占实测（uniform-25 + stream_graph_buckets 基线，
    # 2 warmup + 5 runs 中位数）：chunk-1 hift 16.8→8.9ms、TTFP ~112→~105ms
    # （加不加 hift_compile 都是 ~105）；26 条流式 ASR 转写与基线逐字相同
    # （mean CER 0.1022）。流式中间 chunk 用它可替代 hift_compile（数学等价
    # vs compile 的流式 ~6e-2 漂移，免 15-20s inductor warmup；final chunk /
    # offline 仍走 hift_compile 路径若开）。per-bucket 首遇 lazy capture
    # 数百 ms。质量门 = ASR CER。
    hift_graph_buckets: Optional[str] = None
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
    # [M3.5-r2] ChunkPlanner 参数（默认 = 现行 15/×2 growth 行为）。
    # uniform-25 模式：codec_chunk_frames=25 + codec_chunk_scale=1 → 每 chunk
    # 恒定 25 token（hop 不增长），TTFP 中性、chunk 形状可枚举
    # （+50 mel 帧/chunk，逐 voice 确定），配合 stream_graph_buckets 使用。
    codec_chunk_frames: int = 15
    codec_chunk_scale: int = 2
    # Opt-in Nari-style silent-first suppression.  It gates only the initial
    # PCM, retains pre-roll, and falls back without deleting audio after max_ms.
    trim_leading_silence: bool = False
    leading_silence_preroll_ms: float = 20.0
    leading_silence_max_ms: float = 2000.0
    leading_silence_min_buffer_ms: float = 400.0
