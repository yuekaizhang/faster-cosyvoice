# Faster CosyVoice

English | [简体中文](README_zh.md)

## Overview

Faster CosyVoice is a GPU inference stack for the
[CosyVoice3](https://github.com/QwenAudio/CosyVoice) text-to-speech model. It
accelerates CosyVoice3's two-stage speech-token LLM and DiT Token-to-Wav
pipeline, substantially reducing first-audio latency while increasing
concurrent throughput.

The main optimizations are:

- **DSpark speculative decoding**: a draft model proposes multiple speech
  tokens at once, which the target LLM verifies in parallel to reduce
  autoregressive decoding steps.
- **FlashInfer DiT**: accelerates the DiT model with FlashInfer kernels,
  replacing padded SDPA with ragged attention and fusing QKV, partial RoPE,
  and AdaLN-related operations.
- **Cross-request packed batching**: batches requests while concatenating only
  each request's valid Mel features, avoiding redundant computation on
  padding.
- **Deadline-aware scheduling**: tracks the remaining playable audio for each
  stream in real time. Streams close to underrun are prioritized, as are the
  first audio chunks of newly arrived requests.
- **CUDA Graphs**: reduce kernel-launch overhead for the DiT and HiFT vocoder
  modules.
- **CUDA MPS**: allows the vLLM EngineCore and Token2Wav worker CUDA processes
  to interleave more finely on one GPU, improving overall utilization.

The implementation addresses not only individual model-stage latency, but
also GPU contention between stages, variable-length padding, and streaming
task scheduling. These techniques can also inform other two-stage
"LLM + flow matching" TTS systems.

![Faster CosyVoice architecture](docs/assets/readme/architecture.svg)

## Quick start

### 1. Install

```bash
git clone https://github.com/yuekaizhang/faster-cosyvoice.git
cd faster-cosyvoice
uv sync --frozen
```

### 2. Start the server

```bash
uv run --frozen faster-cosyvoice-server \
  --host 0.0.0.0 \
  --port 8000 \
  --speaker-encoder-tensorrt \
  --streaming-cuda-graph
```

LLM speculative decoding, FlashInfer DiT, cross-request packed batching, and
the deadline-aware scheduler are enabled by default. The command above also
enables TensorRT for the speaker encoder and CUDA Graphs for DiT and the HiFT
vocoder. Streaming starts with a 15-speech-token hop and then grows as
`15 → 30 → 60…`, matching the first-chunk policy used in the performance
benchmarks. `--streaming-cuda-graph` uses built-in, validated graph buckets,
which normally require no manual configuration. See
[advanced configuration](docs/usage.md#advanced-configuration-custom-cuda-graph-buckets)
to tune buckets for a different request-length distribution. CUDA Graphs,
speaker-encoder TensorRT, and CUDA MPS are not enabled automatically; MPS must
be started outside the server process.

### 3. Send a streaming request

```bash
uv run --frozen python examples/stream_client.py \
  --ref-audio ref.wav \
  --ref-text "The exact transcript of the reference audio." \
  --target-text "This is a zero-shot voice-cloning request." \
  --out speech.wav
```

### 4. Run offline inference

```bash
uv run --frozen python examples/offline_inference.py \
  --ref-audio ref.wav \
  --ref-text "The exact transcript of the reference audio." \
  --target-text "This is an offline voice-cloning request." \
  --output-dir results/single
```

### 5. Enable CUDA MPS for additional acceleration

The server creates two CUDA clients: the vLLM EngineCore and the Token2Wav
worker. On a single GPU, kernels from their independent CUDA contexts can
serialize at coarse scheduling boundaries and leave the GPU underutilized.
CUDA MPS routes both processes through a shared GPU scheduling service so
their kernels can interleave or overlap more finely. MPS does not make an
individual kernel faster; the gain depends on how much work from the two
stages can overlap. Start MPS before launching the server:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/fcv-mps/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/fcv-mps/log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

After the MPS daemon starts, launch the server from the same shell. Stop the
daemon after the server exits:

```bash
echo quit | nvidia-cuda-mps-control
```

See [Deployment, API, and configuration](docs/usage.md) for more usage
examples.

## Performance

The table below reports fixed-concurrency measurements on a single NVIDIA
Hopper GPU. Every configuration uses a fixed registered voice, the same set
of reference texts, and equal-length first audio chunks. Each run uses a
15-second warm-up followed by a 60-second measurement window.

| Configuration | Concurrency | TTFP p50 | TTFP p95 | Audio RTFx |
|---|---:|---:|---:|---:|
| Baseline (vLLM + Torch Flow matching) | 1 | 318.8 ms | 392.9 ms | 3.59 |
| Faster CosyVoice | 1 | **72.8 ms** | **82.0 ms** | **15.06** |
| Faster CosyVoice | 8 | 324.3 ms | 391.5 ms | 30.37 |

After optimization, single-concurrency TTFP p50 decreases by 77.2%, while
Audio RTFx increases by 319.7%. See the full [benchmark](docs/benchmark.md)
for additional results.

## Acknowledgments

The model implementation, inference engine, server design, and performance
methodology build on ideas and software from the following open-source
projects. We thank their teams and communities:

- [CosyVoice](https://github.com/QwenAudio/CosyVoice)
- [Nari Qwen3-TTS](https://github.com/nari-labs/nari-qwen3-tts)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [SGLang-Omni](https://github.com/sgl-project/sglang-omni)
