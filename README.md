# Faster CosyVoice

Faster CosyVoice 是面向 NVIDIA GPU 的 CosyVoice3 zero-shot voice-clone
推理与服务实现。LLM 使用 vLLM 和可选 DSpark speculative decoding，token2wav
使用 FlashInfer、跨请求 packed batching、deadline-aware scheduling 和可选 CUDA
Graph。

仓库提供两种入口：

- OpenAI 风格的流式 HTTP API，输出 24 kHz mono PCM16；
- 离线批量推理，支持每条样本独立的 `prompt_audio + prompt_text -> target_text`
  协议。

当前性能配置在单张 NVIDIA H100 80GB 上验证。CUDA Graph 的桶需要按实际音色和文本
长度调优，不能把下面的 H100 推荐值直接视为所有 GPU 的最优值。

## Requirements

- Linux x86_64 容器；
- CUDA 13 兼容的 NVIDIA driver；
- Python 3.12.13（由 `.python-version` 固定）；
- [uv](https://docs.astral.sh/uv/) 0.11.28。

Debian/Ubuntu 容器还需要：

```bash
apt-get update
apt-get install -y build-essential libsndfile1 sox
```

## Install with uv

`pyproject.toml` 与 `uv.lock` 固定了完整依赖，包括带 DSpark patch 的 vLLM fork、
对应的 CUDA 13 预编译扩展以及 TensorRT Python 包。安装只需要一条命令：

```bash
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

## Start the server

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

H100 低延迟配置：

```bash
uv run --frozen faster-cosyvoice-server \
  --speaker-encoder-tensorrt \
  --speech-token-chunk-size 25 \
  --speech-token-chunk-growth 1 \
  --streaming-flow-graph-buckets 512,640,768,896,1024,1280 \
  --streaming-vocoder-graph-buckets 64,128,192,256,384,512 \
  --trim-leading-silence
```

模型、CUDA Graph 和全链路 warmup 完成前，`/ready` 返回 503：

```bash
curl --fail http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/models
```

### CUDA MPS

vLLM EngineCore 与 token2wav worker 是独立 CUDA 进程。同卡部署时，CUDA MPS
允许两个进程的 kernel 并发执行，减少粗粒度时间片抢占：

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d

uv run --frozen faster-cosyvoice-server \
  --speech-token-chunk-size 25 \
  --speech-token-chunk-growth 1 \
  --streaming-flow-graph-buckets 512,640,768,896,1024,1280 \
  --streaming-vocoder-graph-buckets 64,128,192,256,384,512

# 服务退出后关闭 daemon
echo quit | nvidia-cuda-mps-control
```

daemon 只看到一张卡时，client 的 `CUDA_VISIBLE_DEVICES=0` 指可见集合中的第 0 张卡。
自定义 `CUDA_MPS_PIPE_DIRECTORY` 时，路径必须满足 UNIX socket 长度限制。

## Register and use a voice

音色保存在 server 进程内存中，重启后需要重新注册。先注册音色可以避免每个请求都
重复处理参考音频：

```bash
uv run --frozen python examples/register_voice.py \
  --url http://127.0.0.1:8000 \
  --name demo \
  --ref-audio ref.wav \
  --ref-text "Transcript of the reference audio."
```

发送非流式请求：

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

发送流式请求并保存 WAV：

```bash
uv run --frozen python examples/stream_client.py \
  --url http://127.0.0.1:8000 \
  --voice demo \
  --target-text "Streaming speech should start quickly." \
  --out speech.wav
```

客户端同时打印 `ttfb_ms`、`first_playable_ms`、`audible_ttfa_ms`、
`leading_silence_ms` 和 `underruns`。不注册音色时，也可以给请求同时传
`--ref-audio` 与 `--ref-text`。

服务端路由：

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `GET /v1/audio/voices`
- `POST /v1/audio/voices`
- `POST /v1/audio/speech`

`response_format=pcm` 返回 raw little-endian PCM16；`response_format=wav` 的流式
响应使用 unknown-length WAV header。对公网部署前应限制 `ref_audio` 可访问的 URL
与本地路径。

## Offline inference

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

## Performance options

所有 bucket 都采用同一规则：实际长度向上 padding 到第一个足够大的桶，超过最大桶
则回退 eager 路径。Mel 帧率为 50 Hz，所以 64 帧对应 1.28 秒的 Mel，512 帧对应
10.24 秒；Flow 桶覆盖的是包含 prompt 的上下文长度，不等同于纯生成音频时长。

| 参数 | 数值含义 | 生效范围 | 单张 H100 既有观测 |
|---|---|---|---:|
| `--token2wav-cuda-graph-buckets 8,12,16,20,24` | 每个值是 `prompt + generated` 总音频时长，单位秒 | 仅离线 Flow，且 `--token2wav-batch-size 1` | Flow `54 -> 44 ms` |
| `--speaker-encoder-tensorrt` | CampPlus speaker encoder 改用 TensorRT；没有 bucket 值 | 冷 reference frontend；首次构建 engine 约 2–3 分钟 | embedding `58 -> 7 ms` |
| `--vocoder-compile` | HiFT 使用 `torch.compile`，内部自动按 64 Mel 帧取整 | 主要用于离线 HiFT；启动预热约 15–20 秒 | HiFT `48 -> 13 ms` |
| `--streaming-flow-graph-buckets 512,640,768,896,1024,1280` | 每个值是 Flow attention 的完整序列长度，单位 Mel 帧；包含 prompt、首块 padding 和已消费 token | 仅单 session streaming Flow；每个桶首次命中 lazy capture | chunk-1 Flow `89 -> 67 ms` |
| `--streaming-vocoder-graph-buckets 64,128,192,256,384,512` | 每个值是 HiFT 输入长度，单位 Mel 帧 | 仅 streaming 非末块；末块与超长输入回退 eager | chunk-1 HiFT `16.8 -> 8.9 ms` |
| `--speech-token-chunk-size 25 --speech-token-chunk-growth 1` | 首个 hop 消费 25 个 speech token，后续 hop 保持同样大小；speech token 约 25 Hz | 让流式 shape 可枚举，便于稳定命中上面两个 graph 桶 | graph 配套设置 |
| `--token2wav-scheduler deadline` | 同时考虑新请求首块与已播放 stream 的 buffer deadline；`--token2wav-deadline-reserve-ms` 默认 100 | 默认开启 | RPS=1 TTFB p95 `28.1 s -> 201.5 ms`（相对 legacy） |
| `--trim-leading-silence` | 检测并有界裁剪首段静音，默认保留 20 ms pre-roll，并先累计 400 ms startup buffer | 只改变开头 PCM，不减少模型计算 | RPS=1 audible TTFA p95 `905.6 -> 222.4 ms` |

上述 CUDA Graph 与 compile 路径不保证与 eager 波形逐位一致。改变 bucket、精度或
compile 选项后，应使用相同数据集重新验证音频质量。旧版缩写参数仍作为隐藏兼容
别名被接受，但新部署应使用表中的名称。

离线 H100、200 条中文、LLM batch size 16 的参考结果：

| 配置 | LLM token/s | 平均接受长度 | 端到端 RTF |
|---|---:|---:|---:|
| DSpark | 9387.6 | 2.992 | 0.0110 |
| 无 draft | 2057.5 | - | 0.0203 |

这里的 RTF 是 `(LLM wall time + token2wav wall time) / audio duration`，不包含
reference frontend。

## Tests

```bash
uv run --frozen ruff check faster_cosyvoice examples tests
uv run --frozen pytest
uv run --frozen pytest tests/gpu -m gpu -v
```
