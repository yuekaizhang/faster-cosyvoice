# Faster CosyVoice

Faster CosyVoice 是面向 NVIDIA GPU 的 CosyVoice3 zero-shot voice-clone 推理与服务
实现。它使用 vLLM 和可选的 DSpark speculative decoding 生成 speech token，再通过
FlashInfer、跨请求 packed batching 和 deadline-aware scheduling 加速 token2wav。

它同时提供 OpenAI 风格的流式 HTTP API 和离线批量推理，并支持每条请求独立的
`prompt_audio + prompt_text -> target_text` voice-clone 协议。

> [!NOTE]
> 当前功能和性能配置主要在单张 NVIDIA H100 80GB 上验证。其他 GPU、CUDA 版本和
> bucket 组合需要单独验证，下面的性能数字不是对所有环境的保证。

## 核心特性

- OpenAI 风格的 streaming / non-streaming speech API；
- 注册并复用音色，或在单次请求中直接传入参考音频与文本；
- 离线 dataset batching，每条样本使用自己的 prompt audio/text；
- vLLM target model 与可选 DSpark speculative decoding；
- FlashInfer Flow estimator 与跨 session packed token2wav batching；
- deadline-aware scheduler、CUDA Graph、CUDA MPS 等低延迟优化；
- 24 kHz、mono、PCM16/WAV 输出。

## 性能概览

下面的数据来自同一次累积消融测试。完整优化组合相对 vLLM + Torch baseline，在固定
并发 C1 下将 TTFP p50 从 318.8 ms 降至 72.8 ms，p95 从 392.9 ms 降至
82.0 ms；接收音频吞吐从 3.59xRT 提升至 15.06xRT。

| 配置 | 固定并发 | TTFP p50 / p95 | 接收音频吞吐 |
|---|---:|---:|---:|
| vLLM target-only + Torch Flow | C1 | 318.8 / 392.9 ms | 3.59xRT |
| 完整优化组合 + CUDA MPS | C1 | **72.8 / 82.0 ms** | **15.06xRT** |
| 完整优化组合 + CUDA MPS | C8 | 324.3 / 391.5 ms | 30.37xRT |

测试环境为单张 H100 80GB、固定注册音色、Seed-TTS evaluation text、15 秒 warmup
和 60 秒 measurement；使用 closed-loop fixed concurrency、760 ms 首块、不开启
leading-silence trim。TTFP 是客户端收到第一块可播放 PCM 的时间；xRT 是测量窗口内
收到的音频总时长除以窗口时长。详细口径和逐项结果见后文“Benchmark”。

## 快速开始（Quick Start）

下面是一条最短可运行路径：安装依赖、启动默认服务、注册参考音色，然后生成一段 WAV。

### 1. 环境要求

- Linux x86_64 容器；
- CUDA 13 兼容的 NVIDIA driver；
- Python 3.12.13（由 `.python-version` 固定）；
- [uv](https://docs.astral.sh/uv/) 0.11.28。

Debian/Ubuntu 容器还需要：

```bash
apt-get update
apt-get install -y build-essential libsndfile1 sox
```

### 2. 使用 uv 安装

`pyproject.toml` 与 `uv.lock` 固定了完整依赖，包括带 DSpark patch 的 vLLM fork、
对应的 CUDA 13 预编译扩展以及 TensorRT Python 包。安装只需要一条命令：

```bash
git clone https://github.com/yuekaizhang/faster-cosyvoice.git
cd faster-cosyvoice
uv sync --frozen
```

不需要手工 clone vLLM、设置 `PYTHONPATH`、写 `.pth` 或 activate venv。第一次安装会
下载 PyTorch、CUDA 和 vLLM 等大体积依赖；项目内 `.cache/uv/` 会缓存下载内容。

部分基础镜像会预设 `UV_PROJECT_ENVIRONMENT=/opt/...`。如果希望环境始终落在本仓库的
`.venv`，先清掉这个容器级变量：

```bash
unset UV_PROJECT_ENVIRONMENT
uv sync --frozen
```

已有 Hugging Face 缓存时可在安装或启动前设置 `HF_HOME`。模型首次启动会下载默认
LLM、draft model，以及 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 中缺失的 Flow、
HiFT 和 CampPlus 资产。

### 3. 启动服务

`uv run` 会在需要时自动完成同一个 locked environment 的同步，因此也可以直接一条
命令安装并启动：

```bash
uv run --frozen faster-cosyvoice-server --host 0.0.0.0 --port 8000
```

默认开启 DSpark、FlashInfer、跨 session packed batching 和 deadline-aware
scheduler。关闭 speculative decoding：

```bash
uv run --frozen faster-cosyvoice-server --draft-model none
```

模型、CUDA Graph 和全链路 warmup 完成前，`/ready` 返回 503：

```bash
curl --fail http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/models
```

### 4. 注册音色并生成 WAV

准备一段参考音频 `ref.wav` 和它的准确文本。音色保存在 server 进程内存中，重启后
需要重新注册。先注册音色可以避免每个请求都重复处理参考音频：

```bash
uv run --frozen python examples/register_voice.py \
  --url http://127.0.0.1:8000 \
  --name demo \
  --ref-audio ref.wav \
  --ref-text "Transcript of the reference audio."
```

发送流式请求并保存 WAV：

```bash
uv run --frozen python examples/stream_client.py \
  --url http://127.0.0.1:8000 \
  --voice demo \
  --target-text "你好，这是 Faster CosyVoice 生成的语音。" \
  --out speech.wav
```

成功后会生成 `speech.wav`，并在终端打印 TTFB、first playable、
audible TTFA、leading silence 和 underrun 等客户端指标。

## 使用方式

### 注册并复用音色

已注册音色可以通过以下接口查看：

```bash
curl http://127.0.0.1:8000/v1/audio/voices
```

使用已注册的 `demo` 音色发送非流式请求：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "faster-cosyvoice",
    "input": "Hello from Faster CosyVoice.",
    "voice": "demo",
    "response_format": "wav",
    "stream": false
  }' \
  --output speech.wav
```

也可以从 Hugging Face dataset 中选取一行注册音色：

```bash
uv run --frozen python examples/register_voice.py \
  --name benchmark \
  --dataset yuekai/seed_tts_cosy2 \
  --split test_en \
  --index 0
```

### 单次 zero-shot voice clone

不需要先注册音色。下面的 client 会将本地音频编码到请求中：

```bash
uv run --frozen python examples/stream_client.py \
  --ref-audio ref.wav \
  --ref-text "参考音频的准确文本。" \
  --target-text "这是一次不需要注册音色的 zero-shot 请求。" \
  --out zero-shot.wav
```

### HTTP API

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/health` | 进程健康检查 |
| `GET` | `/ready` | 模型与 warmup 就绪检查 |
| `GET` | `/v1/models` | 查看目标模型 |
| `GET` | `/v1/audio/voices` | 查看已注册音色 |
| `POST` | `/v1/audio/voices` | 注册音色 |
| `POST` | `/v1/audio/speech` | 生成语音 |

`POST /v1/audio/speech` 的主要字段：

| 字段 | 说明 |
|---|---|
| `input` | 需要合成的目标文本，必填 |
| `voice` | 已注册音色名；与 `ref_audio + ref_text` 二选一 |
| `ref_audio` | data URI、HTTP(S) URL 或 server 本地路径 |
| `ref_text` | 参考音频的准确文本 |
| `response_format` | `wav` 或 `pcm`，默认 `wav` |
| `stream` | 是否流式返回，默认 `false` |
| `seed` | 采样随机种子，默认 `42` |

`response_format=pcm` 返回 raw little-endian PCM16；流式 WAV 使用
unknown-length WAV header。服务端输出固定为 24 kHz mono。

> [!WARNING]
> 当前 `ref_audio` 允许访问 HTTP(S) URL 和 server 本地路径。对公网部署前，
> 应在网关或应用层增加 URL 白名单、路径限制、认证和请求大小限制。

### 离线推理

单条 voice clone：

```bash
uv run --frozen python examples/offline_inference.py \
  --ref-audio ref.wav \
  --ref-text "参考文本" \
  --target-text "目标文本" \
  --output-dir results/single
```

数据集批量推理：

```bash
uv run --frozen python examples/offline_inference.py \
  --dataset yuekai/seed_tts_cosy2 \
  --split wenetspeech4tts \
  --batch-size 8 \
  --output-dir results/wenetspeech4tts
```

每条输出都使用该行自己的 `prompt_audio` 和 `prompt_text`。输出目录包含 WAV、
`input_manifest.jsonl`、`expected.json`、`run_config.json` 与 `metrics.json`。关闭
speculative decoding 用 `--draft-model none`；FlashInfer 不可用时用
`--flow-estimator torch`。

## 部署配置

### 默认配置

Quick Start 中的命令是功能默认值：DSpark 开启、FlashInfer Flow、
packed token2wav batch size 8、deadline scheduler、`15 × 2` 增长式 speech-token
chunk。CUDA Graph、MPS、speaker TensorRT 和 leading-silence trim 默认不开启。

FlashInfer 在当前 GPU 环境不可用时，可回退到 Torch Flow：

```bash
uv run --frozen faster-cosyvoice-server \
  --streaming-flow-estimator torch \
  --token2wav-batch-mode serial
```

Torch Flow 不支持 packed token2wav，因此两个参数需要一起修改。

### H100 低延迟配置

以下配置使用固定 speech-token hop，并为 streaming Flow 和 HiFT 开启
bucketed CUDA Graph：

```bash
uv run --frozen faster-cosyvoice-server \
  --speaker-encoder-tensorrt \
  --speech-token-chunk-size 25 \
  --speech-token-chunk-growth 1 \
  --streaming-flow-graph-buckets 512,640,768,896,1024,1280 \
  --streaming-vocoder-graph-buckets 64,128,192,256,384,512
```

`--speaker-encoder-tensorrt` 主要降低未缓存参考音频的 frontend 延迟；已注册
音色的每次生成不会重复运行 speaker encoder。各 graph bucket 第一次命中时
会 lazy capture，部署时应在接收流量前做与真实 shape 一致的 warmup。

如果目标是降低 audible TTFA，可额外添加 `--trim-leading-silence`。该选项会修改
开头 PCM，但不减少模型计算量；默认保留 20 ms pre-roll，并在首次发送前
累计至少 400 ms 可播放音频。

### CUDA MPS

vLLM EngineCore 和 token2wav worker 是两个 CUDA 进程。同卡部署时，CUDA MPS
可以让它们的 kernel 更细粒度地共享 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/fcv-mps/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/fcv-mps/log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

uv run --frozen faster-cosyvoice-server \
  --speech-token-chunk-size 25 \
  --speech-token-chunk-growth 1 \
  --streaming-flow-graph-buckets 512,640,768,896,1024,1280 \
  --streaming-vocoder-graph-buckets 64,128,192,256,384,512

# 服务退出后停止同一个 MPS daemon
echo quit | nvidia-cuda-mps-control
```

daemon 只看到一张卡时，client 进程的 `CUDA_VISIBLE_DEVICES=0` 指该可见集合内的
第 0 张卡。`CUDA_MPS_PIPE_DIRECTORY` 应使用较短路径，避免超过 UNIX socket
路径限制。MPS 的收益取决于 GPU、kernel 组合和进程拓扑，不应默认推广到
其他硬件。

## 工作原理

```text
reference audio + reference text
             │
             ├── CampPlus / reference frontend ── condition
             │
target text ─┴── vLLM target + optional DSpark draft
                         │ speech tokens
                         ▼
              deadline-aware token2wav scheduler
                         │
                  FlashInfer Flow
                         │ Mel
                         ▼
                    HiFT vocoder
                         │
                  streaming PCM/WAV
```

| 优化 | 解决的问题 | 主要生效位置 |
|---|---|---|
| DSpark speculative decoding | 减少 target LLM decode step | speech-token 生成 |
| FlashInfer Flow | 降低 Flow/DiT attention 开销 | token2wav |
| Packed batching | 合并多个 session 的 token2wav 工作 | 并发吞吐 |
| Deadline-aware scheduler | 兼顾新请求首块和已播放 stream 的 buffer deadline | 尾延迟与 underrun |
| Flow / HiFT CUDA Graph | 减少固定 shape 的 kernel launch 开销 | streaming token2wav |
| HiFT `torch.compile` | 编译并复用 bucketed vocoder graph | 离线 HiFT |
| Speaker TensorRT | 加速 CampPlus embedding | 冷 reference frontend |
| CUDA MPS | 减少 LLM 与 token2wav 进程的粗粒度抢占 | 同卡多 CUDA client |
| Leading-silence trim | 减少用户感知的开头静音 | audible TTFA |

## 性能调优

### 服务端核心参数

| 参数 | 默认值 | 作用与注意事项 |
|---|---:|---|
| `--gpu-memory-utilization` | `0.5` | vLLM GPU memory budget；需为 token2wav 和 DSpark 留出空间 |
| `--draft-model` | `yuekai/cosyvoice3_llm_dspark` | DSpark draft model；传 `none` 关闭 speculative decoding |
| `--streaming-flow-estimator` | `flashinfer` | 可回退为 `torch` |
| `--token2wav-batch-mode` | `packed` | 跨 session 合并 token2wav；Torch Flow 只能用 `serial` |
| `--token2wav-batch-size` | `8` | packed sub-batch 上限；更大不一定更快 |
| `--token2wav-scheduler` | `deadline` | `legacy` 仅用于兼容和对照实验 |
| `--token2wav-deadline-reserve-ms` | `100` | 已播放 stream 距离 buffer 耗尽的优先级保护窗口 |
| `--speech-token-chunk-size` | `15` | 首个 hop 消费的 speech token 数 |
| `--speech-token-chunk-growth` | `2` | 后续 hop 的增长倍率；固定 shape 时设为 `1` |
| `--trim-leading-silence` | 关闭 | 改变开头 PCM 以降低 audible TTFA，不减少推理计算 |

### CUDA Graph bucket

所有 bucket 都采用同一规则：实际长度向上 padding 到第一个足够大的桶，超过最大桶
则回退 eager 路径。Mel 帧率为 50 Hz，所以 64 帧对应 1.28 秒 Mel，512 帧
对应 10.24 秒 Mel。Flow bucket 覆盖包含 prompt 的完整上下文，不等同于纯生成音频
时长。

| 参数 | bucket 单位 | 生效范围 | 启动/回退行为 |
|---|---|---|---|
| `--token2wav-cuda-graph-buckets 8,12,16,20,24` | `prompt + generated` 总音频秒数 | 离线 Flow，仅 token2wav batch size 1 | 超长回退 eager |
| `--streaming-flow-graph-buckets 512,640,...` | Flow attention 完整序列的 Mel 帧数 | 单 session streaming Flow | 每个 bucket 首次命中 lazy capture |
| `--streaming-vocoder-graph-buckets 64,128,...` | HiFT 输入 Mel 帧数 | streaming 非末块 | 末块和超长输入回退 eager |

### 离线加速选项

| 参数 | 作用 | 开销或限制 | 单张 H100 既有观测 |
|---|---|---|---:|
| `--speaker-encoder-tensorrt` | CampPlus 改用 TensorRT | 首次构建 engine 约 2–3 分钟 | embedding `58 -> 7 ms` |
| `--vocoder-compile` | HiFT 使用 `torch.compile`，输入按 64 Mel 帧取整 | 启动 warmup 约 15–20 秒 | HiFT `48 -> 13 ms` |
| `--token2wav-cuda-graph-buckets` | 离线 Flow CUDA Graph | 仅 token2wav batch size 1 | Flow `54 -> 44 ms` |

上述微基准只说明已测 H100 环境中的量级，不能直接相加或推广到其他 shape。
CUDA Graph 与 compile 路径不保证与 eager 波形逐位一致；改变 bucket、精度或
compile 选项后，应使用同一数据集重新验证音频质量。旧版缩写参数仍作为隐藏
兼容别名被接受，新部署应使用本文中的完整名称。

## 基准测试（Benchmark）

### 指标定义

| 指标 | 定义 |
|---|---|
| TTFP | 从发起请求到客户端收到第一块可播放 PCM；raw PCM 下与 TTFB 一致 |
| Audible TTFA | 在 TTFP 基础上计入首块内的 leading silence 和播放时间 |
| xRT | 测量窗口内接收的音频总时长 / wall-clock 时长；`1xRT` 等于实时速度 |
| RTF | 某段计算时间 / 生成音频时长；数值越小越快 |
| Underrun | 模拟立即播放时，下一块音频未在已有 buffer 耗尽前到达 |

Fixed concurrency 是 closed-loop 负载：每个 worker 等待当前请求完成后再发下一条。
Poisson RPS 是 open-loop 负载：客户端依时钟持续注入请求。两者的横轴数字不能直接
等同；服务过载时，open-loop 的排队和尾延迟会持续增长。

### H100 累积消融

测试时间为 2026-09-02：单张 H100 80GB、固定注册音色、Seed-TTS evaluation
text sequence、15 秒 warmup + 60 秒 measurement、closed-loop fixed concurrency、
不开启 leading-silence trim。消融为累积配置，每行包含之前所有优化。

为保持各后端的首块可比性，该测试使用
`--speech-token-chunk-size 20 --speech-token-chunk-growth 1`，所有实测首块都为
760 ms。这是 benchmark 对齐设置，与前文通用 H100 配置中的 uniform-25 用途不同。

| Step | 累积配置 | C1 TTFP p50 / p95 | C1 xRT | C8 TTFP p50 / p95 | C8 xRT |
|---|---|---:|---:|---:|---:|
| A0 | vLLM target-only + Torch Flow | 318.8 / 392.9 ms | 3.59 | 522.0 / 778.6 ms | 3.67† |
| A1 | A0 + DSpark | 267.4 / 325.3 ms | 3.68 | 480.2 / 718.0 ms | 3.80† |
| A2 | A1 + FlashInfer | 146.0 / 159.7 ms | 10.10 | 216.8 / 243.2 ms | 10.98† |
| A3 | A2 + packed batching | 137.7 / 148.7 ms | 10.17 | 565.0 / 664.7 ms | 18.76 |
| A4 | A3 + deadline scheduler | 137.4 / 150.8 ms | 10.28 | 563.2 / 669.9 ms | 18.83 |
| A5 | A4 + Flow CUDA Graph | 121.0 / 127.8 ms | 12.39 | 564.2 / 676.0 ms | 19.11 |
| A6 | A5 + HiFT CUDA Graph | 106.5 / 113.3 ms | 13.43 | 490.3 / 617.4 ms | 21.59 |
| A7 | A6 + CUDA MPS | **72.8 / 82.0 ms** | **15.06** | **324.3 / 391.5 ms** | **30.37** |

A0–A6 关闭 MPS，A7 只在 A6 基础上开启 MPS。C1 所有点均完整返回且没有
underrun。带 † 的 A0–A2 C8 虽然完整返回 PCM，但分别出现 14、12、13 条
underrun，因此其 raw xRT 仅作透明性记录，不应解读为质量安全的服务容量。
A3–A7 在 C1 和 C8 都没有 underrun。

从 A0 到 A7，C1 TTFP p50 降低 77.2%，p95 降低 79.1%，xRT 提高
319.7%。最后一项 MPS（A6 → A7）使 C1 p50 降低 31.7%，C8 xRT 提高
40.7%。增量收益与加入顺序和 workload 相关，不能将每项百分比独立相加。

deadline scheduler 在稳定 closed-loop C1/C8 测试中的平均收益很小；它主要保护
bursty/open-loop 流量下的首块和已开始播放的 stream deadline。该性能消融没有测量
语义语音质量。

### 离线参考结果

单张 H100、200 条中文、LLM batch size 16：

| 配置 | LLM token/s | 平均接受长度 | 端到端 RTF |
|---|---:|---:|---:|
| DSpark | 9387.6 | 2.992 | 0.0110 |
| 无 draft | 2057.5 | — | 0.0203 |

这里的 RTF 是 `(LLM wall time + token2wav wall time) / audio duration`，不包含
reference frontend。

## 兼容性与限制

- 当前主要验证 Linux x86_64、CUDA 13 和单张 H100 80GB；
- vLLM 依赖指向锁定 commit 的 DSpark patch fork，并复用其 base commit 对应的
  CUDA 13 预编译扩展；
- server 输出固定为 24 kHz mono PCM16/WAV；
- 注册音色只保存在当前 server 进程内存中，重启后需要重新注册；
- streaming Flow graph 仅覆盖配置 bucket 内的 shape，超出后自动回退 eager；
- MPS 需要 driver 和容器允许运行 MPS daemon，而且只对适合的多 CUDA client
  拓扑有意义；
- graph/compile 可能导致波形不再逐位一致，上线前应重新进行音频质量回归。

## 常见问题

| 现象 | 处理方式 |
|---|---|
| `uv` 将环境安装到 `/opt/...` | 执行 `unset UV_PROJECT_ENVIRONMENT`，再运行 `uv sync --frozen` |
| `/ready` 持续返回 503 | 查看 server log；首次模型下载、TensorRT build 和 graph capture 可能耗时 |
| FlashInfer 环境检查失败 | 改用 `--streaming-flow-estimator torch --token2wav-batch-mode serial` |
| GPU OOM | 降低 `--gpu-memory-utilization` 或 `--token2wav-batch-size`，并减少 graph bucket |
| CUDA Graph 没有加速 | 检查实际 shape 是否超出最大 bucket，并确认已对目标 shape 预热 |
| MPS 没有收益 | 确认 LLM 和 token2wav 进程均接入同一 MPS server，并与 MPS OFF 做 A/B |
| 启用 trim 后开头波形改变 | 这是预期行为；需要保留原始 PCM 时关闭 `--trim-leading-silence` |

查看完整 CLI 帮助：

```bash
uv run --frozen faster-cosyvoice-server --help
uv run --frozen python examples/offline_inference.py --help
```

## 开发与测试

```bash
uv run --frozen ruff check faster_cosyvoice examples tests
uv run --frozen pytest
uv run --frozen pytest tests/gpu -m gpu -v
```

默认 pytest 不执行标记为 `gpu` 的测试。GPU 测试会加载完整模型，运行前应确认
当前 GPU 没有被其他服务占用。

## 致谢

本项目建立在 [CosyVoice](https://github.com/QwenAudio/CosyVoice)、
[vLLM](https://github.com/vllm-project/vllm)、
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 等项目之上。延迟和播放指标的
定义参考 [Nari Labs benchmarks](https://github.com/nari-labs/benchmarks)。

## 许可证

当前仓库尚未添加顶层 `LICENSE` 文件。对外分发前需要明确本项目许可证，并同时
遵守 CosyVoice、模型权重以及各依赖的许可条款。
