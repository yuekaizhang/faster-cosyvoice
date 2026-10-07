# Faster CosyVoice 使用指南

## 在线推理

在线请求可以直接携带参考音频，也可以先在服务端注册音色，再通过名称复用。

### 请求级 zero-shot voice clone

`stream_client.py` 会将本地参考音频编码到当前请求中：

```bash
uv run --frozen python examples/stream_client.py \
  --ref-audio ref.wav \
  --ref-text "参考音频的准确文本。" \
  --target-text "这是需要合成的目标文本。" \
  --out speech.wav
```

### 注册并复用音色

音色缓存在 server 进程内，重启后需要重新注册：

```bash
uv run --frozen python examples/register_voice.py \
  --url http://127.0.0.1:8000 \
  --name demo \
  --ref-audio ref.wav \
  --ref-text "参考音频的准确文本。"
```

注册后，通过音色名称发送流式请求：

```bash
uv run --frozen python examples/stream_client.py \
  --url http://127.0.0.1:8000 \
  --voice demo \
  --target-text "你好，这是 Faster CosyVoice 生成的语音。" \
  --out speech.wav
```

也可以从 Hugging Face dataset 中选择样本注册：

```bash
uv run --frozen python examples/register_voice.py \
  --name benchmark \
  --dataset yuekai/seed_tts_cosy2 \
  --split test_en \
  --index 0
```

查询当前服务进程中已注册的音色：

```bash
curl http://127.0.0.1:8000/v1/audio/voices
```

### HTTP API

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/health` | 进程健康检查 |
| `GET` | `/ready` | 模型和 warmup 就绪检查 |
| `GET` | `/v1/models` | 查看当前加载的模型 |
| `GET` | `/v1/audio/voices` | 查看已注册音色 |
| `POST` | `/v1/audio/voices` | 注册音色 |
| `POST` | `/v1/audio/speech` | 生成语音 |

`POST /v1/audio/speech` 的主要字段：

| 字段 | 说明 |
|---|---|
| `input` | 目标文本，必填 |
| `voice` | 已注册音色；与 `ref_audio + ref_text` 二选一 |
| `ref_audio` | data URI、HTTP(S) URL 或 server 本地路径 |
| `ref_text` | 参考音频的准确文本 |
| `response_format` | `wav` 或 `pcm`，默认 `wav` |
| `stream` | 是否流式返回，默认 `false` |
| `seed` | 采样随机种子，默认 `42` |

使用已注册音色发送非流式请求：

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

`response_format=pcm` 返回 raw little-endian PCM16；流式 WAV 使用 unknown-length
WAV header。服务端固定输出 24 kHz 单声道音频。

## 离线推理

单条推理：

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

每条样本都使用各自的 `prompt_audio` 和 `prompt_text`。输出目录包含 WAV、
`input_manifest.jsonl`、`expected.json`、`run_config.json` 和 `metrics.json`。

常用消融参数：

```bash
# 关闭 DSpark
--draft-model none

# 使用 Torch Flow
--flow-estimator torch
```

## 性能配置

### Streaming chunk 与 CUDA Graph

推荐配置使用固定 25-token speech-token hop，并为 streaming Flow 和 HiFT 开启
bucketed CUDA Graph。固定 hop 可以让请求 shape 更稳定，从而提高 graph bucket
命中率。

`--speaker-encoder-tensorrt` 只影响未缓存参考音频的 frontend；已注册音色不会为每个
请求重复运行 speaker encoder。Graph bucket 第一次命中时会 lazy capture，部署时应
使用接近真实 shape 的请求预热。

### CUDA Graph bucket

实际长度会向上 padding 到第一个足够大的 bucket；超过最大 bucket 时回退到 eager
执行。Speech token 的帧率约为 25 Hz，Mel 帧率为 50 Hz，即一个 speech token
对应约两个 Mel frame。因此，64 个 Mel frame 对应 1.28 秒，512 个 Mel frame 对应
10.24 秒。

Flow bucket 覆盖包含 prompt 在内的完整上下文，还可能包含 lookahead 和 padding，
不能直接视为新生成音频的时长。

| 参数 | 单位 | 生效范围 |
|---|---|---|
| `--token2wav-cuda-graph-buckets 8,12,...` | 总音频秒数 | 离线 Flow，Token2Wav batch size 1 |
| `--streaming-flow-graph-buckets 512,640,...` | Mel frame | 单 session streaming Flow |
| `--streaming-vocoder-graph-buckets 64,128,...` | Mel frame | streaming 非末块 HiFT |

### Leading silence trim

如需降低 audible TTFA，可以开启 `--trim-leading-silence`。它会裁剪播放端收到的开头
静音 PCM，但不会减少模型计算，也不会降低模型侧 TTFP。

### 核心参数

下表记录 CLI 的内置默认值。

| 参数 | 内置默认值 | 作用 |
|---|---:|---|
| `--gpu-memory-utilization` | `0.5` | vLLM 显存预算；需给 Token2Wav 和 DSpark 留空间 |
| `--draft-model` | `yuekai/cosyvoice3_llm_dspark` | 传 `none` 关闭 speculative decoding |
| `--streaming-flow-estimator` | `flashinfer` | 可回退为 `torch` |
| `--token2wav-batch-mode` | `packed` | Torch Flow 只能使用 `serial` |
| `--token2wav-batch-size` | `8` | packed sub-batch 上限 |
| `--token2wav-scheduler` | `deadline` | `legacy` 用于兼容和对照 |
| `--token2wav-deadline-reserve-ms` | `100` | buffer 接近耗尽时的保护窗口 |
| `--speech-token-chunk-size` | `15` | 第一个 streaming hop 的 token 数 |
| `--speech-token-chunk-growth` | `2` | 后续 hop 的增长倍率；固定 shape 设为 `1` |
| `--streaming-flow-graph-buckets` | 未设置 | streaming Flow CUDA Graph bucket |
| `--streaming-vocoder-graph-buckets` | 未设置 | streaming HiFT CUDA Graph bucket |
| `--speaker-encoder-tensorrt` | 关闭 | 用 TensorRT 运行 CampPlus speaker encoder |
| `--trim-leading-silence` | 关闭 | 修改开头 PCM，降低 audible TTFA |

查看完整 CLI：

```bash
uv run --frozen faster-cosyvoice-server --help
uv run --frozen python examples/offline_inference.py --help
```
