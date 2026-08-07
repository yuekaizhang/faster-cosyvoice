# faster-cosyvoice

CosyVoice3 加速 voice-clone 推理：vLLM + DSpark 投机解码（LLM）+ FlashInfer DiT（token2wav）。
设计文档：docs/superpowers/specs/2026-08-05-faster-cosyvoice-design.md

## 环境（容器内）
    bash scripts/setup_env.sh
    source venv/bin/activate
    export PYTHONPATH=$PWD/third_party/spec-vllm:$PYTHONPATH
    export HF_HOME=...   # 可选，指向已有缓存

环境为分层复用：vllm==0.25.1 wheel（dspark 内置）等重依赖来自 vllm025_venv（.pth 挂载），
详见 requirements.txt 注释与 scripts/setup_env.sh。

## Offline 推理
    # 单条
    python examples/offline_inference.py --ref-audio ref.wav \
        --ref-text "参考文本" --target-text "目标文本" --output-dir results/single
    # 数据集批量（wenetspeech4tts 是 split）
    python examples/offline_inference.py --dataset yuekai/seed_tts_cosy2 \
        --split wenetspeech4tts --batch-size 8 --output-dir results/ws4tts
    # 关投机解码 / torch estimator 降级
    --draft-model none / --estimator torch

脚本内部 `OMP_NUM_THREADS=1`（vLLM EngineCore fork 后 OpenMP 会 segfault），可用环境变量覆盖。

## 测试
    pytest                     # CPU 单测（gpu/integration 默认跳过）
    pytest tests/gpu -m gpu -v # offline e2e + server e2e + interleave 回归（容器内）
    python scripts/asr_check.py --wav-dir results/... --ref-json results/.../expected.json \
        --paraformer-dir models/sherpa-onnx-paraformer-zh-2023-09-14

## 基准（H100，yuekai/seed_tts_zh_cosy3 test_zh 200 条，bs16）

复现命令：

    python examples/offline_inference.py --dataset yuekai/seed_tts_zh_cosy3 \
        --split test_zh --limit 200 --batch-size 16 --output-dir results/bench_dspark
    python examples/offline_inference.py --dataset yuekai/seed_tts_zh_cosy3 \
        --split test_zh --limit 200 --batch-size 16 --draft-model none \
        --output-dir results/bench_baseline

| 配置 | llm tok/s | 平均接受长度 | 端到端 RTF |
|---|---|---|---|
| dspark | 9387.6 | 2.992 | 0.0110 |
| baseline（无 draft） | 2057.5 | — | 0.0203 |

llm 加速比 4.56x。注意：SpeechSpec 参考值为 bs16 下 1.70x（全量 2020 条）。本仓数字更高的可能原因：
仅 200 条（continuous batching 尾部效应不同）、0.5B 模型 bs16 下 per-step 开销占主导使步数减少收益
放大、计时口径差异（本仓只计 llm.generate 墙钟）。接受长度 2.992 与参考值 2.98 一致，说明投机
解码行为本身对齐。
RTF 为 (llm+t2w)/音频时长，不含 frontend。

### 性能旋钮（opt-in，默认全关；26 条 batch=1 实测，H100）

| 旋钮 | 作用 | 收益 |
|---|---|---|
| `--t2w-cuda-graph-buckets "8,12,16,20,24"` | batch=1 offline flow 分桶 CUDA graph | flow 54→44ms |
| `--campplus-trt` | campplus 走 TensorRT（plan 磁盘缓存，首次构建 ~90s） | campplus 58→7ms |
| `--hift-compile` | hift torch.compile + 64 帧长度桶（init warmup ~15-20s） | hift 48→13ms |

三项全开：batch=1 每请求 e2e 301→219ms（26 条总墙钟 ~7s，RTF ~0.032）；ASR 门均通过。
注意：`--hift-compile` 下波形与 eager 非逐位一致（offline ~8e-4，流式 worst-chunk ~6e-2），
需要逐位稳定时保持关闭；batch≥8 时 flow 走 packed 批量，graph 桶自动不生效。

流式 TTFP 旋钮（server，opt-in，默认关；单并发 H100 实测，同一 cached voice
2 warmup + 5 runs 中位数）：

| 旋钮 | 作用 | 收益 |
|---|---|---|
| `--stream-graph-buckets "512,640,768,896,1024,1280"` | 单 session 流式 flow 分桶 CUDA graph（mel 帧） | chunk-1 flow 89→67ms |
| `--codec-chunk-frames 25 --codec-chunk-scale 1` | uniform-25 chunk（形状可枚举，配合上行） | — |
| `--hift-graph-buckets "64,128,192,256,384,512"` | [M3.5-r4] 流式中间 chunk hift 整段分桶 CUDA graph（数学等价 ~3e-4） | chunk-1 hift 16.8→8.9ms，TTFP −7~8ms |

两项 + `--hift-compile`：TTFP 160→149ms（无 LLM 抢占时 flow 89→23ms；流式 flow 与
LLM 解码同卡并发，压缩空间被 SM 争抢部分吃掉）。graph 内 dense-SDPA 与 eager
flashinfer 非逐位一致（mel corr ~0.985+；26 条流式 ASR mean CER 0.1127 vs offline
基线 0.1078），需要逐位稳定请保持关闭。每 bucket 首遇 lazy capture ~25-100ms；
首 chunk 长度 ≈ (prompt+pad+25)×2 帧（逐 voice 确定），建议在典型值附近配细桶。
r4 复测（HEAD，uniform-25 + stream-graph-buckets 基线，同法）：TTFP 112.8ms；
`+hift-compile` 111.6ms；`+hift-graph-buckets` 104.4-104.6ms（26 条流式 ASR
转写与基线逐字相同，mean CER 0.1022）。高优先级 CUDA stream（priority=-1 包
t2w）实测 TTFP 无变化（vLLM EngineCore 为独立进程/上下文，stream 优先级仅
在同一 context 内生效），未收录。

### CUDA MPS（同卡部署最大 TTFP 杠杆，-28ms）

vLLM EngineCore 与 token2wav 是两个进程，默认时间片轮转共享 GPU——流式 chunk
与 LLM decode 并发时互相整片抢占。启用 MPS 后两进程 kernel 合并进同一 context
并发执行：同卡 TTFP 104.6→**76.1ms**（config C+hift-graph；不开 hift-graph 时
77.9ms——MPS 下抢占消失，hift-graph 增益缩小到 ~2ms）。ASR 转写与非 MPS 逐字
相同（数值不受影响）。用法（server 与 EngineCore 都会继承 env 成为 MPS client）：

    export CUDA_VISIBLE_DEVICES=<gpu>          # daemon 侧选卡
    nvidia-cuda-mps-control -d                 # 默认 pipe /tmp/nvidia-mps
    CUDA_VISIBLE_DEVICES=0 python -m faster_cosyvoice.server.app ...  # client 内索引从 0 起
    echo quit | nvidia-cuda-mps-control        # 结束后务必关闭 daemon

注意：MPS client 的 `CUDA_VISIBLE_DEVICES` 是 daemon 可见集合内的索引（daemon
绑单卡时 client 用 0）；自定义 `CUDA_MPS_PIPE_DIRECTORY` 路径须 <108 字符
（UNIX socket 限制，过长 daemon 会静默退出）。

## Streaming server（M2+M3）

    bash scripts/run_server.sh --port 8000          # 启动（含引擎加载+warmup，~2分钟）
    # 流式请求（客户端示例，打印 TTFA/时长）：
    python examples/stream_client.py --url http://localhost:8000 \
        --ref-audio ref.wav --ref-text "参考文本" --target-text "目标文本" --out out.wav
    # curl（非流式）：
    curl -X POST http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
        -d '{"input":"你好","voice":"my_voice"}' -o out.wav
    # 音色注册 / 列表：
    curl -X POST http://localhost:8000/v1/audio/voices -H 'Content-Type: application/json' \
        -d '{"name":"my_voice","ref_audio":"data:audio/wav;base64,...","ref_text":"参考文本"}'
    curl http://localhost:8000/v1/audio/voices

协议：OpenAI `/v1/audio/speech`（扩展 `ref_audio`/`ref_text`/`seed`），`response_format` wav|pcm，
`stream:true` 为增量 PCM（首包 44 字节未知长度 WAV 头）。流式 token2wav 默认走 flashinfer
estimator（chunk-causal custom mask，fp16）+ 跨 session packed 批量（`--t2w-batch-mode packed`，
batcher v2）；回退开关：`--stream-estimator torch --t2w-batch-mode serial`（M2 torch 路径，
两者需同时指定）。LLM 侧 DSpark 投机解码照常生效。
音色注册为内存态，重启即失。`--draft-model none` 关投机解码；`--gpu-memory-utilization` 默认 0.5。

实测（H100，e2e 测试 4 并发流式，同一提交顺序跑两轮）：

| 配置 | server TTFA (ms) | client TTFA (ms) | 每请求 wall (s) |
|---|---|---|---|
| torch + serial（M2，回退） | 483 / 674 / 865 / 1059 | 923–1499 | 2.2–3.1 |
| flashinfer + packed（M3，默认） | 350 / 468 / 463 / 466 | 640–753 | 0.8–1.1 |

ASR 门（CER≤0.15）流式/非流式 5/5 通过（flashinfer+packed per-item CER 0.06–0.11），
同 seed 流式与非流式转写一致。交错回归测试锁定多 session 确定性
（tests/gpu/test_interleave.py、test_stream_flashinfer.py、test_stream_batched.py）。

M1 offline + M2 streaming server + M3（flashinfer 流式 mask + 跨请求 packed batch）全部落地。
