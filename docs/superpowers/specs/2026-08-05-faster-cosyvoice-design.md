# faster-cosyvoice — CosyVoice3 加速 voice-clone 推理设计

- 日期：2026-08-05
- 状态：设计已逐段评审通过，待写实现计划
- 范围：仅 CosyVoice3（Fun-CosyVoice3-0.5B-2512）、仅 zero-shot voice clone

## 1. 目标与非目标

**目标**

1. **Offline 批量推理脚本**：`--ref-audio/--ref-text/--target-text` 单条模式；`--dataset/--split/--batch-size` 数据集模式（如 `yuekai/seed_tts_cosy2` 的 wenetspeech4tts 子集，26 条 prompt/text 对）。每批内先生成该批全部 speech token，再批量 token2wav（批粒度由 `--batch-size` 控制，批间串行）。
2. **Streaming server**：OpenAI `/v1/audio/speech` 协议（`ref_audio`/`ref_text` 扩展字段），流式 WAV/PCM 输出，服务端跨请求组 batch（v1：LLM continuous batching + token2wav asyncio 交错；v2/M3：token2wav 跨请求 packed batch），不用 Triton Inference Server。
3. **LLM 加速**：vLLM + DSpark draft 投机解码（SpeechSpec 方案，target/draft 均从 HF 加载）。
4. **token2wav 加速**：FlashInfer 加速的 DiT estimator（duplex/CosyVoice fork 的实现），offline 用 packed varlen 批量路径。
5. **独立仓库**：所需 CosyVoice 代码最小集拷入本仓库；权重全部从 HF 拉取。

**非目标（v1 不做）**

- CosyVoice2、SFT/instruct 模式、跨语言指令
- mp3 编码、变速（speed）、流式文本输入（bistream）
- hift / campplus 的批量化（保持逐条循环，draft 明确允许）
- Triton Inference Server / 多进程部署、Prometheus 指标（只做结构化日志 + `/health`）

## 2. 参考源与取用清单

| 参考源 | 取什么 | 关键路径 |
|---|---|---|
| SpeechSpec | prompt 构建、speculative_config 组装、采样参数、指标读取、环境配方 | `/lustre/fsw/portfolios/coreai/users/yuekaiz/speculative/SpeechSpec/{README_TTS.md, benchmark_tts.py, run_cv3_speculative.sh}` |
| duplex/CosyVoice fork | token2wav 全栈：`CosyVoice3_Token2Wav` 基线 + `FlashInferDiT` + packed 批量路径 + torch 流式路径（`forward_stream`）；cosyvoice 最小代码集 | `/lustre/fsw/portfolios/coreai/users/yuekaiz/duplex/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3{,_flashinfer}.py` 及 `cosyvoice/`、`third_party/Matcha-TTS/matcha/utils/audio.py` |
| triton BLS 方案 | 流式状态机逻辑、首块优先级方案、`_prepare_prompt` 前端处理 | 同 fork `runtime/triton_trtllm/model_repo_cosyvoice3/cosyvoice3/1/model.py` |
| vllm-omni cosyvoice3 分支 | chunker 数学、OpenAI speech 协议层片段、TTFP 优化经验（进程级组件缓存、s3tokenizer GPU torch 模型、campplus TRT plan 磁盘缓存）、voice/speaker 缓存设计 | `/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/tmp/vllm-omni-cosyvoice/vllm_omni/{model_executor/stage_input_processors/cosyvoice3.py, entrypoints/openai/serving_speech.py}` |
| qi-hua/async_cosyvoice | 流式音频编码（未知长度 WAV 头）、自适应 chunk 思想、voice 注册接口形态；**不作为代码基础**（仅 CV2、vllm 0.7.3、并发稳定性差） | GitHub |

**模型与数据资产（HF）**

- LLM target：`yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF`（0.5B Qwen2 系，speech token 为词表内 `<|s_N|>` 字面 token）
- LLM draft：`yuekai/cosyvoice3_llm_dspark`（DSpark 并行 drafter，3 层，block_size 8 → num_speculative_tokens 7，8192 缩减 draft 词表 + d2t 映射）
- token2wav 权重：`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（`cosyvoice3.yaml`、`flow.pt`、`hift.pt`、`campplus.onnx`、`CosyVoice-BlankEN` tokenizer 目录）
- 评测数据：`yuekai/seed_tts_cosy2`（wenetspeech4tts 子集等；其 token 列是 cosy2 的，对 CV3 不可直接用）、`yuekai/seed_tts_zh_cosy3`（带 `prompt_audio_cosy3_tokens` 列，可作 fast path）

## 3. 关键决策

**D1 架构路线：全新单进程 asyncio server（非剥离 vllm-omni、非双进程）。**
理由：vllm-omni 的 talker 是自定义 vLLM 模型（embedding 拼接 + 模型内 RAS sampler），spec decode 未接线且 RAS 与 draft rejection-sampling 冲突——按本设计换成 SpeechSpec 方案后该 stage 整体作废；两 stage 间的 SharedMemory connector / runner / orchestrator（15k+ LOC）在单进程直调下全部不需要。"剥离"做到最后等于"往新骨架拷文件"。双进程（`vllm serve` + 编排层）则让 token 级流式多一跳 SSE、TTFA 变差、offline/online 路径分叉。

**D2 LLM = plain HF checkpoint + 标准 vLLM + DSpark，不用 RAS、不注册自定义 vLLM 模型。**
prompt 用 chat template `continue_final_message` 续写 `<|s_N|>`；标准采样（见附录 A）SpeechSpec 已验证质量与加速（bs1 1.93x → bs64 1.18x，平均接受长度 ~2.98）。spec decode 做成开关（`--draft-model none` 关闭），文档标注高并发（bs≥64）收益降至 ~1.18x。

**D3 流式 estimator 分阶段（用户已选定）。**
`FlashInferDiT` 现只支持 offline（`assert streaming=False`）。机理澄清：**已播出音频的不可变性由 mel cache + `speech_offset` 后缀切片保证，与 attention mask 无关**；chunk-causal attention 的作用是让全前缀重算与已缓存 mel 在 chunk 边界保持一致，并与流式训练图数值对齐（duplex fork 的 torch 流式路径：`streaming=True` → `add_optional_chunk_mask`，static_chunk_size=50，配合固定 `rand_noise` 缓冲实现确定性重算；cosyvoice3.yaml 将 static_chunk_size=chunk_size×token_mel_ratio=50 接进 DiT，`yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX` 中自导出的 streaming ONNX 图同理）。注意：vllm-omni 实践中流式实际跑的是全注意力图（streaming 标志在 cfm 层被丢弃）且每 chunk 重采噪声，质量可用——chunk-causal 并非硬前提，但 v1 仍选 fork 的 torch streaming 路径（最贴近训练图、重算确定）。方案：**v1 流式用 torch streaming estimator（chunk-causal + 固定 rand_noise），offline 从第一天用 flashinfer packed batch；v2 给 FlashInferDiT 实现 chunk-causal custom mask（flashinfer ragged prefill 支持 custom mask），以 v1 torch 流式输出为参照 + ASR 门通过后切为流式默认。** 推论：v1 流式的跨请求 token2wav 并发 = asyncio 交错；真正的跨请求 packed batch 随 v2 落地（batcher 接口不变，只换执行器）。

**D4 prompt speech token 现场计算。**
token2wav 本来就需要 ref audio（mel + campplus），统一用 s3tokenizer `speech_tokenizer_v3_25hz` GPU torch 模型现算 v3 token（比捆绑 ONNX CPU 路径快 ~30x，vllm-omni 经验）；数据集若带 `prompt_audio_cosy3_tokens` 列则作为 fast path 直接采用。因此任何带（audio, text, target_text）的数据集都能跑，不依赖预算 token 列。

**D5 环境固化。**
（实施修正，2026-08-05：`vllm-omni==0.25.1` 在 PyPI 不存在，且 DSpark 实现在 **vllm 0.25.1 wheel 本体**（`vllm/v1/worker/gpu/spec_decode/`），vllm-omni 并非依赖。）实际配方：分层复用现有 `tts/vllm025_venv`（vllm==0.25.1 wheel + torch/flashinfer 0.6.13/triton/s3tokenizer/x-transformers 等全部重依赖）——本仓 venv（py3.12，uv 管理）通过 `.pth` 挂载其 site-packages，仅补装 datasets/soundfile/pytest；`yuekaizhang/vllm@dspark-draft-sampling-mirrors` 源码盖 PYTHONPATH（rep-penalty mirror，PR #48932 合并前必需；`.so` 与 `_version.py` 从 wheel 软链进源码树）。写进 `scripts/setup_env.sh`，服务启动时做版本/符号自检。

**D6 单 GPU 默认共存。**
vLLM `gpu_memory_utilization`：server 0.5 / offline 0.8 可配；token2wav 同卡常驻（~3–4GB）；`--token2wav-device` 可选分卡。

**D7 质量门 = ASR（CER），不是张量 diff。**
fp16/flashinfer 路径与参考实现张量级偏差 10–20% rel 是已知良性现象；所有优化以 sherpa-onnx paraformer ASR（现成 transcribe-tts-output 流程）在固定测试集上的 CER 为回归门。

## 4. 总体架构与仓库布局

单进程。offline 脚本与 streaming server 是两个薄入口，共享同一套核心模块。

```
faster-cosyvoice/
├── faster_cosyvoice/
│   ├── config.py            # 所有旋钮：chunk 尺寸、设备、spec decode、采样参数、批量上限
│   ├── assets.py            # HF 下载/校验三个模型 repo；缺失时报确切 repo id 与目标路径
│   ├── llm/
│   │   ├── prompt.py        # SpeechSpec build_prompts/speech_id_str/PUNCTS 迁移
│   │   ├── engine.py        # vllm.LLM / AsyncLLM 工厂 + speculative_config 组装（读 draft config.json）
│   │   └── sampling.py      # CV3 标准采样参数（附录 A）
│   ├── token2wav/           # 独立子包：不 import vLLM，可单独测试
│   │   ├── cosyvoice/       # 从 duplex fork 拷的最小代码集（附录 B）
│   │   ├── matcha_audio.py  # mel_spectrogram（附录 A 参数）
│   │   ├── frontend.py      # RefAudioFrontend：s3tokenizer(GPU)/campplus/prompt mel/2:1 截断
│   │   │                    #   + 进程级组件缓存 + speaker cache（LRU，key 见 §5.2）
│   │   ├── token2wav.py     # CosyVoice3Token2Wav：offline_batch()（flashinfer packed）
│   │   │                    #   + stream_step()（v1 torch 流式路径）
│   │   └── flashinfer_dit.py# FlashInferDiT + packed varlen 批量；v2 加 chunk-causal mask
│   ├── streaming/
│   │   ├── chunker.py       # ChunkPlanner 纯数学状态机（无 I/O）
│   │   ├── session.py       # StreamSession：累计 token、mel cache、speech_offset、chunk_index
│   │   └── batcher.py       # Token2WavWorker：优先队列 + 执行器（v1 串行 / v2 packed batch）
│   ├── server/
│   │   ├── app.py           # FastAPI lifespan：自检 → 加载引擎 → warmup → 接流量
│   │   ├── openai_speech.py # /v1/audio/speech、/v1/audio/voices
│   │   └── audio_encode.py  # 未知长度 WAV 头 + PCM 增量编码
├── examples/offline_inference.py
├── scripts/{setup_env.sh, run_server.sh}
├── requirements.txt
└── tests/                   # CPU 单测 + GPU 集成测试（§8）
```

## 5. 组件设计

### 5.1 `llm/engine.py`

- `create_offline_llm(cfg) -> vllm.LLM` 与 `create_async_llm(cfg) -> AsyncLLM`，共享 speculative_config 组装：从 draft 目录 `config.json` 读 `speculators_model_type` → method、`block_size - 1` → num_speculative_tokens，`draft_sample_method='probabilistic'`，rp≠1.0 时 `draft_apply_repetition_penalty=True`。
- 输出提取不 detokenize：启动时扫词表建 `token_id → speech_id` 查表数组（`<|s_N|>` → N，其余 -1）；offline 批量与 streaming DELTA 增量共用。终止用 checkpoint 的 `eos_token_id`（158486 = `<|eos1|>`）或显式 `stop_token_ids=[158486]`——**不可用 stop 字符串**：vLLM 在 `detokenize=False` 时会直接拒绝 stop strings。

### 5.2 `token2wav/` 子包

- **`frontend.py`**：`RefAudioFrontend.process(wav_16k, ref_text) -> RefCondition{prompt_speech_tokens, prompt_feat, spk_embedding}`；`process_batch()` 批量版（s3tokenizer 批量、campplus/mel 逐条）。flow 用 2:1 截断后的 (feat, token) 对，LLM 用未截断完整 token。speaker cache：key = `voice_name` 或 `sha256(音频字节)+ref_text`（修正 triton 版按 ref_text 碰撞、无淘汰的缺陷），LRU 有上限。进程级单例缓存 tokenizer/特征提取器/campplus（vllm-omni TTFP 经验：省 ~450ms/请求）。
- **`token2wav.py`**：`CosyVoice3Token2Wav(model_dir, device, estimator_mode)`。构建：`load_hyperpyyaml(cosyvoice3.yaml, overrides=...)` 只实例化 flow+hift，overrides 至少 `{'llm': None, 'hifigan': None}`（参照 fork `cosyvoice/flow/flow.py` `__main__` 的做法；duplex 的 token2wav_cosyvoice3.py 只 override 了 qwen_pretrain_path、会连带拉起 Qwen2 与 GAN 判别器，此处修正），实现时按 yaml 顶层键核对，原则 = llm/判别器等训练节点全部置 None；`flow.pt` strict 加载、`hift.pt` 去 `generator.` 前缀。两个入口：
  - `offline_batch(tokens_list, cond_list) -> list[wav]`：flashinfer packed varlen（即 fork 的 `token2wav_forward_batched`），内部按 `--token2wav-batch-size` 分段防 OOM；hift 逐条。
  - `stream_step(session, cumulative_tokens, finalize) -> pcm`：v1 torch 路径 = `flow.inference(streaming=True)` 全前缀重算 → 按 `token_offset×2` 切新 mel → 拼 session mel cache → hift 重跑全量 mel 按 `speech_offset` 切新音频。chunk 粒度完全由 ChunkPlanner 驱动（fork `forward_stream` 的内部循环及其常数 25/×2/100 不采用，只取其单步计算逻辑）；固定 `rand_noise` 缓冲保证重算确定性。模块本身无状态，多请求交错安全。
  - `estimator_mode ∈ {flashinfer, torch, flashinfer_streaming(v2)}`。
- **`flashinfer_dit.py`**：拷贝 `FlashInferDiT`（`BatchPrefillWithRaggedKVCacheWrapper`、fused qkv/adaLN、partial RoPE、plan 缓存）+ packed 批量函数。注意：packed 不等长批量依赖 OpenAI Triton 编译器（pip `triton`，源码有 `_HAS_TRITON` assert），环境自检覆盖。CUDA graph 模式 v1 不启用（packed eager 已优于 TRT）。

### 5.3 `streaming/`

- **`chunker.py`**：`ChunkPlanner(prompt_token_len)`；输入 `(available_tokens, llm_finished)` 输出 `ChunkPlan{prefix_len, token_offset, finalize} | None`。常数：首块 = 15 + pad(prompt 至 15 倍数) + 3 lookahead；hop 逐块 ×2 封顶 60（15/60 取自 vllm-omni `deploy/cosyvoice3.yaml` 的 `codec_chunk_frames: 15`、封顶=默认 4×chunk；vllm-omni 代码默认值是 25/100，勿混淆）；LLM 结束后余量一次 finalize（哪怕不足一个 hop）。纯函数，黄金序列单测。
- **`batcher.py`**：单 asyncio worker 独占 token2wav GPU 调用；`submit(ChunkJob) -> awaitable pcm`；优先队列 `(chunk_index, 到达序)`（首块插队，保 TTFA 公平——triton priority 方案的 asyncio 版）。v1 执行器逐 job 串行；v2 每轮弹出所有分属不同 session 的 ready job → 一次 packed forward → 按 session 拆分。同 session 的 chunk 因 mel cache 依赖严格串行，天然不同批。

### 5.4 `server/`

- `POST /v1/audio/speech`：`{model, input, voice | (ref_audio: URL/base64, ref_text), response_format: wav|pcm, stream, seed}`。stream=true → StreamingResponse（首块前发未知长度 WAV 头）；stream=false → 同管线拼接整段返回。`max_tokens = min(2048, 20 × text_token_len)`。
- `POST /v1/audio/voices`：注册音色 = 跑一次 frontend 存入 speaker cache，返回 voice_name。
- `app.py` lifespan：环境自检 → assets 校验 → 加载 AsyncLLM + token2wav → warmup（flashinfer plan、campplus TRT plan、一条 dummy 请求全链路）→ 开始接流量。

### 5.5 `examples/offline_inference.py`

单条与数据集两种模式（§6 offline 数据流）。指标输出：LLM tok/s、draft 接受率/平均接受长度（`llm.get_metrics()` 的 `vllm:spec_decode_*` 计数器）、t2w ms/sample、端到端 RTF（25 token/s 音频常数）。

## 6. 数据流与时序

### 6.1 Offline（数据集模式，`--batch-size B`）

```
load_dataset → [{ref_wav, ref_text, target_text}] → 按 B 切片，逐批：
  ① frontend.process_batch(B 条 ref_wav)      # s3tokenizer GPU 批量；campplus/mel 逐条
  ② build_prompts(ref_text+target_text, ①的 token)
  ③ llm.generate(B 条, 每条独立 seed)          # 批内 continuous batching + DSpark
  ④ 查表提取 <|s_N|> → tokens_list
  ⑤ token2wav.offline_batch(tokens_list, conds) # flashinfer packed，内部子批
  ⑥ 落盘 wav + 指标
```

批间串行（LLM 与 token2wav 共卡，跨批流水线留作后续优化）。单条模式即 B=1。

### 6.2 Streaming（单请求）

```
POST /v1/audio/speech(stream=true)
  ① 音色解析：voice → cache 命中（~0ms）；否则 frontend.process
  ② build prompt → AsyncLLM.generate(DELTA, detokenize=False, stop_token_ids=[eos1])
  ③ token 消费循环：delta → 查表 → session 缓冲 → 问 ChunkPlanner
  ④ ChunkPlan → submit ChunkJob → await pcm
  ⑤ pcm → audio_encode（首块前 WAV 头）→ yield HTTP chunk
  ⑥ LLM 结束 → 余量 finalize=True → hift finalize → 关流
```

关键性质：③ 与 ④⑤ 解耦——chunk 在 GPU 执行期间消费循环继续攒 token，下一 chunk 自然拿到更大 hop（与 15→30→60 翻倍节奏吻合）。TTFA 分解（热身后）：音色缓存命中 ≈0 + LLM prefill+首块所需 18–32 token（15+pad(0–14)+3，spec decode 下几十 ms）+ 首 chunk flow+hift（~50–100ms）→ 目标与 vllm-omni 实测同量级（~200ms 热路径）。

### 6.3 并发行为

- LLM：AsyncLLM 原生跨流 continuous batching。
- token2wav v1：优先队列 + 逐 job 串行，跨流靠交错；每流 hop 翻倍使人均调用频率随时长下降。v2：跨 session packed batch。
- 背压/内存：HTTP 写侧背压只挡音频发送；token 生成上限 `max_tokens≤2048`（~80s 音频）封顶每 session 缓冲。断连 → abort LLM 请求 + 丢弃排队 job + 释放 session。

## 7. 错误处理与部署

**启动期 fail fast**：assets 缺失报确切 repo id；环境自检（patched vllm 的 rp-mirror 符号、vllm==0.25.1、flashinfer/s3tokenizer/triton 可导入）不过则带修复指引退出；warmup 失败即启动失败。campplus TRT plan 按 device+TRT 版本落盘缓存。

**请求期错误只杀本请求**：入参校验 400（ref_audio 解码失败/缺 ref_text/空 input；ref 统一重采样 16k mono，>30s 截断并警告）；`finish_reason=length` 照常合成但计数告警；LLM 结束且累计有效 speech token 为 0 → 500（只要有非零余量就按 §5.3 finalize 出短音频）；token2wav 异常/OOM 由 batcher 捕获、只 fail 该 job 的 future，worker 存活；v2 packed cap 批大小。offline 模式逐条记录失败不中断整批。全局每请求超时旋钮。

**降级开关**：`--draft-model none`；`--estimator torch`（flashinfer 不可用时 offline 也可跑）；campplus TRT→ORT-CPU 自动回退。

**部署**：`setup_env.sh` 固化环境配方；`run_server.sh` 供容器内直接跑（Slurm：login node 只做规划/git，计算在容器内）。可观测性 v1 = 结构化日志（每请求 TTFA/chunk 数/接受率）+ `/health`。

## 8. 测试与验证

**CPU 单测（login node 可跑）**：ChunkPlanner 黄金序列；prompt 构建与 SpeechSpec `build_prompts` 输出逐字节对齐（本地 CosyVoice-BlankEN tokenizer fixture）；token 查表提取；WAV 流式头；batcher 优先级/取消语义（fake executor）；speaker cache key/LRU。

**GPU 集成/回归（容器内）**：
1. 质量门 = ASR：sherpa-onnx paraformer（transcribe-tts-output 流程）对 wenetspeech4tts 26 条测 CER；flashinfer offline、torch streaming、v2 flashinfer streaming 三条路径过同一道门。
2. 流式一致性：同一请求 stream/非 stream 都过 ASR；chunk 拼接样本数连续无缝。
3. E2E 基准：offline 脚本 draft 开/关对照；验收 = 复现 SpeechSpec LLM 加速（bs1 ~1.9x、接受长度 ~3.0）与 duplex estimator 批量数字（b8 ~1.35 ms/sample，同硬件同量级）。
4. Server 冒烟：并发 N 客户端测 TTFA/RTF（协议与 vllm-omni 相同，现有 online benchmark 客户端小改可用）。

## 9. 里程碑（每个 = 独立实现计划 + PR）

- **M1 offline**：仓库骨架 + `assets.py` + D5 环境脚本/启动自检 + `llm/` 子包（prompt/engine/sampling + DSpark speculative_config）+ token2wav 拷贝/flashinfer + offline 脚本 + ASR 门 → 单条与数据集批量可用。§8 测试 1 的 offline 路径与测试 3（draft 开/关对照）的验收归 M1。
- **M2 streaming v1**：torch 流式 estimator + chunker/batcher/OpenAI 协议/voice 注册 → 并发流式可用，TTFA 与参考实现同量级。
- **M3 streaming v2**：FlashInferDiT chunk-causal custom mask + 跨请求 packed batch，ASR 门通过后切流式默认。

## 10. 风险与未知项

1. **flashinfer 流式 mask（M3 核心风险）**：chunk-causal custom mask 的正确性需 GPU 上以 torch streaming 输出为参照 + ASR 验证；plan/mask 内存与缓存策略需实测。M3 失败不影响 M1/M2 可用性。
2. **hift 全量重跑 O(T²)**：流式路径 hift 每 chunk 重跑累计 mel；hop 翻倍部分摊销，长输出（>30s）下的尾部延迟需实测，必要时限制单请求时长或引入 hift 增量 cache（CosyVoice 上游有 mel/source cache + fade 方案可参考）。
3. **环境脆弱**：PYTHONPATH 覆盖 wheel 的方式对 vllm 升级敏感；PR #48932 合并后应尽快去掉 overlay。flashinfer/triton 版本未在参考源钉死，plan() 签名随版本变化——requirements.txt 里显式钉。
4. **spec decode 高并发收益递减**（bs64 仅 1.18x）：server 高并发场景可能需要默认关 draft；做成运行时配置并在文档写明。
5. **s3tokenizer 版本**：确认 pip 版本支持 `speech_tokenizer_v3_25hz` GPU torch 模型加载。
6. **数据集 token 列陷阱**：`yuekai/seed_tts_cosy2` 的 `prompt_audio_cosy2_tokens` 是 CV2 token，对 CV3 无效——已由 D4（现场计算）规避，脚本忽略 cosy2 列。

## 附录 A — 关键常数

| 项 | 值 |
|---|---|
| LLM 采样 | temperature 0.8, top_p 0.95, top_k 15, repetition_penalty 1.1, max_tokens=min(2048, 20×text_len), 每请求独立 seed |
| speculative_config | method dspark, num_speculative_tokens 7 (block 8), draft_sample_method probabilistic, draft_apply_repetition_penalty True |
| prompt 格式 | chat [user: "You are a helpful assistant.<\|endofprompt\|>"+ref_text+target_text（去 PUNCTS 引号括号）, assistant: 连接的 "<\|s_id\|>"]，apply_chat_template(continue_final_message=True)；终止 = checkpoint eos_token_id 158486（`<\|eos1\|>`）；detokenize=False 下用 stop_token_ids，不可用 stop 字符串 |
| token 速率 | 25 speech token/s；token_mel_ratio 2（50 mel fps）；24kHz 输出（hop 480） |
| chunker | 首块 hop 15 + pad(prompt→15 倍数) + lookahead 3；×2 增长封顶 60 |
| mel | n_fft 1920, win 1920, hop 480, 80 bins, 24kHz, fmin 0, **fmax None**, center False（matcha mel_spectrogram） |
| flow | 10 步 Euler，cosine t-scheduler，inference_cfg_rate 0.7，CFG 双行（packed 时 2B 行） |
| DiT | dim 1024 × 22 层 × 16 头 × 64；partial RoPE 仅首 64 通道（head 0） |
| campplus | kaldi fbank 80 维 @16k、去均值 → 192 维 |
| ref 音频 | 输入 16kHz mono；prompt feat/token 2:1 截断给 flow，完整 token 给 LLM |
| GPU 显存 | server：vLLM gpu_memory_utilization 0.5；offline：0.8；token2wav 常驻 ~3–4GB |

## 附录 B — cosyvoice 代码拷贝清单（自 duplex fork）

- `runtime/triton_trtllm/token2wav_cosyvoice3.py`（基线，改造为 `token2wav.py`：yaml overrides、去掉 TRT estimator 分支——estimator_mode 只有 {flashinfer, torch}；campplus 的 TRT 加速保留）
- `runtime/triton_trtllm/token2wav_cosyvoice3_flashinfer.py` → `flashinfer_dit.py`
- `cosyvoice/flow/flow.py`（**trim 模块级 `cosyvoice.utils.onnx` import**）、`flow_matching.py`
- `cosyvoice/flow/DiT/dit.py`、`DiT/modules.py`
- `cosyvoice/hifigan/generator.py`、`f0_predictor.py`
- `cosyvoice/transformer/upsample_encoder.py`（整文件拷贝，仅 trim `class_utils` 相关 import 连锁，与文件级对应原则一致）、`convolution.py` 及其被 hift/f0_predictor 依赖的少量文件
- `cosyvoice/utils/mask.py`、`common.py`
- `third_party/Matcha-TTS/matcha/utils/audio.py`（mel）；matcha `BASECFM` 基类 ~40 行直接内联，避免拖入 matcha decoder/pylogger
- 第三方 pip：torch、torchaudio、flashinfer、triton（OpenAI Triton 编译器，packed 批量必需）、x_transformers、einops、hyperpyyaml、omegaconf、onnxruntime、s3tokenizer、librosa、scipy

拷贝原则：保持文件级对应（便于 diff 上游），只做 import 路径与上述 trim 修改；改动处注释标注来源 commit。
