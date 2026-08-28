# Faster CosyVoice

## TL;DR

Faster CosyVoice 是面向 NVIDIA H100 的 CosyVoice3 zero-shot voice-clone
推理与服务实现：LLM 使用 vLLM + DSpark speculative decoding，token-to-wave
使用 FlashInfer、跨请求 packed batching 和可选 CUDA Graph。

它同时提供离线批量推理和 OpenAI 风格的流式 HTTP API。服务端输出 24 kHz mono
PCM16；音色可先注册并复用，也可在单次请求中携带参考音频。

> [!NOTE]
> **性能口径**
>
> 仓库原有的 TTFP 是首个 PCM chunk 延迟。新增的 benchmark 直接复用
> [Nari Labs tts-bench](https://github.com/nari-labs/benchmarks)，以 Poisson
> open-loop 负载测量客户端 TTFB、first playable、**audible TTFA**、leading
> silence 和播放 underrun。两种指标不能混为一谈。完整对照见
> [Nari 加速方案评估](docs/nari-acceleration-review.md)。

## Requirements

- Linux x86_64 容器；当前验证硬件为 NVIDIA H100 80GB
- CUDA 13 兼容的 NVIDIA driver
- [uv](https://docs.astral.sh/uv/)
- Python 3.12.13（由 `.python-version` 固定）
- 一个预装 vLLM 0.25.1、PyTorch/CUDA、FlashInfer 和 CosyVoice 运行依赖的
  base venv；默认位置是相邻目录 `../vllm025_venv`

Debian/Ubuntu 容器中的系统包：

```bash
apt-get update
apt-get install -y build-essential libsndfile1 sox
```

模型首次启动可能从 Hugging Face 下载 LLM、draft model 和缺失的 Codec 资产。
已有缓存时设置 `HF_HOME`，可以避免重复下载。

## Setup with uv

`pyproject.toml` 和签入的 `uv.lock` 固定项目 Python 依赖。CUDA 重依赖继续复用
`VLLM_BASE_VENV`，因此不会在项目 venv 中重复安装整套 PyTorch/vLLM。

```bash
export VLLM_BASE_VENV=/path/to/vllm025_venv  # 默认 ../vllm025_venv
export HF_HOME=/path/to/existing/huggingface/cache  # 可选

bash scripts/setup_env.sh
source scripts/env.sh
source "$FCV_VENV/bin/activate"
faster-cosyvoice-server --help
```

`scripts/env.sh` 把 FlashInfer、Numba、Triton 和 TorchInductor 的运行缓存放在
项目 `.cache/` 下，并兼容以前创建的 `venv/`。`uv sync --frozen` 是有意为之：
lockfile 过期时直接失败，而不是静默解析一套不同环境。

## Start the HTTP server

默认 profile 使用 DSpark、FlashInfer 和跨 session packed batching：

```bash
bash scripts/run_server.sh --host 127.0.0.1 --port 8000
```

低首音频延迟的 H100 配置：

```bash
bash scripts/run_server.sh \
  --host 127.0.0.1 --port 8000 \
  --campplus-trt \
  --codec-chunk-frames 25 --codec-chunk-scale 1 \
  --stream-graph-buckets 512,640,768,896,1024,1280 \
  --hift-graph-buckets 64,128,192,256,384,512 \
  --trim-leading-silence
```

token-to-wave 默认使用 Nari-style deadline-aware scheduling：新请求首块优先，
但已经开始播放且即将耗尽 buffer 的 stream 会按最早 deadline 抢占，避免持续流被
首块洪峰饿死。`--trim-leading-silence` 是可选的 audible-TTFA 优化：它按 benchmark
同一检测规则裁掉首段静音，保留 20 ms pre-roll，并在首次发送前累计至少 400 ms
可播放音频以避免紧接着 underrun。需要保留原始开头 PCM 时不要打开该参数。

模型加载、CUDA Graph capture 和一条全链路 warmup 完成前，`/ready` 返回 503：

```bash
curl --fail http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/models
```

### CUDA MPS

vLLM EngineCore 和 token-to-wave worker 是独立 CUDA 进程。在同一张 H100 上启用
MPS 可以避免粗粒度时间片抢占。仓库此前的单请求内部首 PCM chunk 中位数为
`104.6 ms -> 76.1 ms`；这不是 Nari audible TTFA。

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d
bash scripts/run_server.sh \
  --codec-chunk-frames 25 --codec-chunk-scale 1 \
  --stream-graph-buckets 512,640,768,896,1024,1280 \
  --hift-graph-buckets 64,128,192,256,384,512

# 服务退出后关闭 daemon
echo quit | nvidia-cuda-mps-control
```

daemon 只看到一张卡时，client 的 `CUDA_VISIBLE_DEVICES=0` 指的是该可见集合内的
第 0 张卡。自定义 `CUDA_MPS_PIPE_DIRECTORY` 时路径需短于 UNIX socket 限制。

## Register a reusable voice

音色存于当前 server 进程内存，重启后需要重新注册。推荐把注册放到部署启动流程，
这样每个 TTS 请求不再重复运行参考音频解码和 CampPlus。

```bash
python examples/register_voice.py \
  --url http://127.0.0.1:8000 \
  --name demo \
  --ref-audio ref.wav \
  --ref-text "Transcript of the reference audio."

curl http://127.0.0.1:8000/v1/audio/voices
```

也可直接从缓存的数据集取一行作为固定 benchmark voice：

```bash
python examples/register_voice.py \
  --name benchmark \
  --dataset yuekai/seed_tts_cosy2 \
  --split test_en --index 0
```

## API

服务提供：

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `GET /v1/audio/voices`
- `POST /v1/audio/voices`
- `POST /v1/audio/speech`

### Non-streaming request

`POST /v1/audio/speech` 使用 OpenAI Audio Speech 请求形状，并扩展支持
`ref_audio`、`ref_text`、`language` 和 `seed`。

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF",
    "input": "Hello from Faster CosyVoice.",
    "voice": "demo",
    "language": "English",
    "response_format": "wav",
    "stream": false
  }' \
  --output speech.wav
```

### Streaming request and audible timing

```bash
python examples/stream_client.py \
  --url http://127.0.0.1:8000 \
  --voice demo \
  --target-text "Streaming speech should start quickly." \
  --out speech.wav
```

客户端请求原始 PCM 流并打印 Nari-compatible 的 `ttfb_ms`、
`first_playable_ms`、`audible_ttfa_ms`、`leading_silence_ms` 和 `underruns`。
也可以用 `--ref-audio` 与 `--ref-text` 做单次 zero-shot clone。

`response_format=pcm` 返回 raw little-endian PCM16；`response_format=wav` 在流式
响应开头返回 unknown-length WAV header。采样率固定为 24 kHz mono。

## Nari-compatible benchmark

安装相邻 `../nari/benchmarks/tts_bench` 的原版锁定环境，并准备其固定的 1,088 条
英文 Seed-TTS prompt：

```bash
bash scripts/setup_nari_benchmark.sh
```

启动服务并注册名为 `benchmark` 的固定 voice 后，每个 RPS 点独立运行：

```bash
RPS=1 SEED=0 WARMUP=30s DURATION=5m \
  bash scripts/run_nari_benchmark.sh

RPS=6 SEED=0 WARMUP=30s DURATION=5m \
  bash scripts/run_nari_benchmark.sh
```

脚本固定使用完整文本 HTTP、raw mono PCM16/24 kHz、Poisson open-loop arrival、
120 秒 request timeout 和最多 4096 个 in-flight request。结果写入
`benchmarks/results/`，每个目录包含原始请求、到达序列、音频、`summary.json` 和
`report.txt`。没有 `DEEPGRAM_API_KEY` 时不测 WER，延迟和 underrun 不受影响。
提交性能数字时使用上面的 30 秒 warmup / 5 分钟 measurement；快速调参可临时用
`WARMUP=15s DURATION=60s`，但应明确标为 scout。

### 本容器的 5 分钟结果

单张 H100 80GB + CUDA MPS，使用上面的低延迟配置、固定 benchmark voice、
seed 0；每个点 30 秒 warmup + 5 分钟 measurement。raw PCM 的 TTFB 与 first
playable 相同，可视为“首个可播放 PCM/TTFP”；audible TTFA 还包含播放到首个可听声
所需的时间。

| 请求 RPS（实际） | 请求数 | TTFB p50 / p95 / p99 | audible TTFA p50 / p95 / p99 | E2E p95 | 完整返回 / underrun |
|---:|---:|---:|---:|---:|---:|
| 1 (1.073) | 322 | 84.2 / 186.7 / 214.8 ms | 104.2 / 206.7 / 234.8 ms | 444.7 ms | 322/322 / 0 |
| 6 (5.853) | 1,756 | 218.1 / 422.3 / 558.7 ms | 238.1 / 442.5 / 582.8 ms | 1551.9 ms | 1,756/1,756 / 0 |

RPS=1 的 322 条全部可听。RPS=6 有 1,755/1,756 条达到 audible threshold；剩余
一条虽然完整返回 11.2 秒 PCM，但整段低于 -45 dBFS 检测阈值，因此本轮的 transport /
capacity 成功率是 100%，语义质量门仍是 fail。没有配置 Deepgram key，所以不能报告
WER。原始报告见 `benchmarks/results/canonical-deadline-trim-buffer400-rps-*/report.txt`。

### 容量 frontier 的 60 秒 scout

单张 H100 80GB + CUDA MPS，使用上面的低延迟配置、seed 0。TTFP 在不同项目中
含义不统一，因此这里同时列客户端收到首个可播放 PCM 的 TTFB 和 Nari 定义的
audible TTFA：

| 配置 | 请求 RPS（实际） | TTFB p50 / p95 | audible TTFA p50 / p95 | 成功率 | underrun 请求 |
|---|---:|---:|---:|---:|---:|
| deadline-aware | 1 (1.117) | 80.8 / 201.5 ms | 341.3 / 905.6 ms | 100% | 0% |
| deadline-aware | 6 (6.350) | 241.2 / 416.3 ms | 469.9 / 1023.0 ms | 100% | 0% |
| deadline-aware | 8 (8.200) | 334.7 / 452.1 ms | 570.2 / 1123.5 ms | 100% | 0% |
| deadline-aware | 10 (9.817) | 7738.9 / 12735.5 ms | 7985.9 / 13120.0 ms | 77.42% | 94.91% |
| + silence trim / 400ms buffer | 1 (1.117) | 85.0 / 202.4 ms | 105.0 / 222.4 ms | 100% | 0% |
| + silence trim / 400ms buffer | 6 (6.350) | 276.2 / 520.3 ms | 296.2 / 540.3 ms | 100% | 0% |

旧 `(chunk_index, arrival)` 调度用完全相同的 RPS=1 arrival sequence 时，TTFB
p95 是 28.119 秒、40.30% 请求 underrun。deadline-aware 修复将其降到
201.5 ms 且零 underrun。当前配置的容量边界位于 8–10 RPS；RPS=8 虽然保持
连续播放，E2E p95 已到 5.19 秒，RPS=10 则明确过载。以上是单 seed scout，完整
原始结果位于本机 `benchmarks/results/`。上面的 5 分钟结果给出稳定点，frontier 表仍是
单 seed scout；正式容量结论还需要 RPS=8/10 的 5 分钟、多 seed 和 WER。

## Offline inference

```bash
# 单条
python examples/offline_inference.py \
  --ref-audio ref.wav \
  --ref-text "参考文本" \
  --target-text "目标文本" \
  --output-dir results/single

# 数据集批量
python examples/offline_inference.py \
  --dataset yuekai/seed_tts_cosy2 \
  --split wenetspeech4tts \
  --batch-size 8 \
  --output-dir results/wenetspeech4tts
```

关闭 speculative decoding 用 `--draft-model none`；FlashInfer 不可用时用
`--estimator torch`。脚本默认 `OMP_NUM_THREADS=1`，避免 vLLM EngineCore fork 后
OpenMP runtime 冲突。

## Performance notes

离线 H100、200 条中文、batch size 16 的既有结果：

| 配置 | LLM token/s | 平均接受长度 | 端到端 RTF |
|---|---:|---:|---:|
| DSpark | 9387.6 | 2.992 | 0.0110 |
| 无 draft | 2057.5 | - | 0.0203 |

DSpark 的 LLM 阶段加速为 `4.56x`。RTF 是 `(LLM + token2wav wall time) /
audio duration`，不包含 reference frontend。

主要 opt-in 旋钮：

| 参数 | 作用 | 既有 H100 观测 |
|---|---|---:|
| `--t2w-cuda-graph-buckets 8,12,16,20,24` | batch=1 offline Flow graph | Flow `54 -> 44 ms` |
| `--campplus-trt` | CampPlus TensorRT | embedding `58 -> 7 ms` |
| `--hift-compile` | offline HiFT compile + bucket | HiFT `48 -> 13 ms` |
| `--stream-graph-buckets ...` | 单 session streaming Flow graph | chunk-1 Flow `89 -> 67 ms` |
| `--hift-graph-buckets ...` | streaming HiFT graph | chunk-1 HiFT `16.8 -> 8.9 ms` |
| deadline-aware scheduler（默认） | 首包与播放 deadline 联合调度 | RPS=1 TTFB p95 `28.1s -> 201.5ms` |
| `--trim-leading-silence` | 有界首段静音抑制 + startup buffer | RPS=1 audible TTFA p95 `905.6 -> 222.4ms` |

CUDA Graph/compile 路径与 eager 并非逐位相同；启用新 bucket 或精度优化时应跑
ASR/CER 质量门。Nari 方案中哪些已覆盖、哪些值得下一步实现，见
[docs/nari-acceleration-review.md](docs/nari-acceleration-review.md)。

## Tests

```bash
source scripts/env.sh
source "$FCV_VENV/bin/activate"
bash scripts/check.sh
python -m pytest tests/gpu -m gpu -v

python scripts/asr_check.py \
  --wav-dir results/... \
  --ref-json results/.../expected.json \
  --paraformer-dir models/sherpa-onnx-paraformer-zh-2023-09-14
```

设计背景与实现计划保存在
`docs/superpowers/specs/2026-08-05-faster-cosyvoice-design.md`。
