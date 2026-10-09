# Faster CosyVoice

[English](README.md) | 简体中文

## 项目介绍

Faster CosyVoice 是 [CosyVoice3](https://github.com/QwenAudio/CosyVoice) 语音合成模型的 GPU
推理方案。针对 CosyVoice3 的 speech-token LLM 与 DiT Token-to-Wav 两阶段推理链，
Faster CosyVoice 采用多种加速手段，大幅降低流式合成的首包延迟，并提升并发吞吐。

主要加速手段包括：

- **DSpark speculative decoding**：draft model 一次提出多个 speech token，由 target
  LLM 并行验证，减少自回归 decode step。
- **FlashInfer DiT**：使用 FlashInfer 算子库对 DiT 模型进行加速，包括将 padded SDPA
  改成 ragged attention，并融合 QKV、partial RoPE 和 AdaLN 相关操作。
- **跨请求 packed batching**：将不同请求拼成 batch 计算，只拼接每条请求的有效 Mel 声学特征，避免
  padding 带来的冗余计算。
- **Deadline-aware scheduling**：请求调度方面，实时跟踪每条 stream 还能播放多久；
  即将耗尽的请求会优先调度，新到请求的首 chunk 音频也会高优处理。
- **CUDA Graph**：对 DiT 和 HiFT Vocoder 模块开启 CUDA Graph，减少 kernel launch 开销。
- **CUDA MPS**：让 vLLM EngineCore 与 token2wav worker 两个 CUDA 进程在同一张卡上
  更细粒度地交错执行，进一步提高 GPU 效率。

这套实现不仅关注单个推理阶段的速度，也处理两个阶段共享 GPU 时产生的资源竞争、
变长请求 padding 和流式任务调度等问题。该实现也可以为其他
“LLM + flow matching”两阶段 TTS 系统提供参考。

![Faster CosyVoice architecture](docs/assets/readme/architecture.svg)

## 快速开始

### 1. 安装

```bash
git clone https://github.com/yuekaizhang/faster-cosyvoice.git
cd faster-cosyvoice
uv sync --frozen
```

### 2. 启动服务

```bash
uv run --frozen faster-cosyvoice-server \
  --host 0.0.0.0 \
  --port 8000 \
  --speaker-encoder-tensorrt \
  --streaming-cuda-graph
```

LLM speculative decoding、FlashInfer DiT、跨请求 packed batching 和
deadline-aware scheduler 是内置默认。上面的命令额外开启 speaker encoder
TensorRT，并为 DiT 和 HiFT Vocoder 开启 CUDA Graph。默认 streaming hop 从 15 个
speech token 开始，随后按 `15 → 30 → 60…` 增长，与性能测试使用的首包策略一致。
`--streaming-cuda-graph` 会使用内置且经过验证的 graph bucket，
通常不需要手动配置；按请求长度分布调优 bucket 的方法见
[高级配置](docs/usage_zh.md#高级设置自定义-cuda-graph-bucket)。CUDA Graph、speaker
encoder TensorRT 和 CUDA MPS 不会默认开启；MPS 需要在服务进程外单独启动。

### 3. 发送流式请求

```bash
uv run --frozen python examples/stream_client.py \
  --ref-audio ref.wav \
  --ref-text "参考音频的准确文本。" \
  --target-text "这是一次 zero-shot voice clone 请求。" \
  --out speech.wav
```

### 4. 离线推理

```bash
uv run --frozen python examples/offline_inference.py \
  --ref-audio ref.wav \
  --ref-text "参考音频的准确文本。" \
  --target-text "这是一次离线 voice clone 请求。" \
  --output-dir results/single
```

### 5. 进一步加速：开启 CUDA MPS

服务启动以后会产生 vLLM EngineCore 和 token2wav worker 两个 CUDA client。同卡部署时
两个独立 CUDA context 的 kernel 容易在较粗的调度边界串行，造成 GPU 空泡。CUDA
MPS 将多进程工作提交给共享的 GPU 调度服务，使 LLM 和 token2wav 的 kernel 能够更细
粒度地交错或并行执行，从而填补空泡、提高整卡利用率。MPS 不会让单个 kernel 本身变快，
实际收益取决于两阶段是否有可重叠的计算。可以在启动服务前开启 MPS：

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/fcv-mps/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/fcv-mps/log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

MPS daemon 启动后，在同一个 shell 中执行上文介绍的服务启动命令。服务退出后关闭
daemon：

```bash
echo quit | nvidia-cuda-mps-control
```

更多用法请参考 [部署、API 与配置](docs/usage_zh.md)。

## 性能

下表来自单张 Hopper GPU 上的固定并发测试。所有配置使用固定注册音色、同一组
参考文本和相同长度首包音频；测试前 warmup 15 秒，measurement 窗口 60 秒。

| 配置 | 并发 | TTFP p50 | TTFP p95 | Audio RTFx |
|---|---:|---:|---:|---:|
| Baseline (VLLM + Torch Flow matching) | 1 | 318.8 ms | 392.9 ms | 3.59 |
| Faster CosyVoice | 1 | **72.8 ms** | **82.0 ms** | **15.06** |
| Faster CosyVoice | 8 | 324.3 ms | 391.5 ms | 30.37 |

经过优化以后的 CosyVoice 模型，单并发 TTFP (p50) 降低 77.2%，Audio RTFx 提高 319.7%。更多测试结果见 [性能测试](docs/benchmark_zh.md)。

## 致谢

本项目的模型实现、推理引擎、服务设计和性能测试方法参考了以下开源项目，感谢相关团队和社区的工作：

- [CosyVoice](https://github.com/QwenAudio/CosyVoice)
- [Nari Qwen3-TTS](https://github.com/nari-labs/nari-qwen3-tts)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [SGLang-Omni](https://github.com/sgl-project/sglang-omni)
