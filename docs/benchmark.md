# Faster CosyVoice Benchmark

English | [简体中文](benchmark_zh.md)

This document defines the Faster CosyVoice performance metrics, reports the
cumulative ablation results, and compares Faster CosyVoice with other
CosyVoice3 inference frameworks.

## Metrics and methodology

| Metric | Definition | Interpretation |
|---|---|---|
| TTFP | Time from sending a request until the client receives the first playable PCM chunk | Lower is better |
| p50 TTFP | Median TTFP across all requests | Typical request latency |
| p95 TTFP | TTFP not exceeded by 95% of requests | Tail latency |
| RTFx | Total audio duration received during the measurement window / measurement-window duration | Higher is better; `1 RTFx` is real-time generation |

This document consistently uses **RTFx** for audio throughput. It is the
inverse direction of the per-request RTF metric: RTF divides compute time by
audio duration and is better when lower, whereas RTFx measures generated audio
per unit of wall time and is better when higher.

Fixed concurrency `C` means that `C` workers run simultaneously. Each worker
waits for its current request to finish before sending the next request, so the
x-axis represents the number of continuously active requests. The measurement
protocol is:

- one NVIDIA Hopper GPU;
- fixed reference audio and reference text;
- a 15-second warm-up followed by a 60-second measurement window;
- seed `0`;
- leading-silence trimming disabled.

## Faster CosyVoice cumulative ablation

All ablation measurements use a concurrency of one. Every request completes
without an underrun, and every first audio chunk is 760 ms long. A0 through A5
run without MPS; A6 adds MPS to A5. Each row includes all optimizations from
the preceding rows.

| Step | Cumulative configuration | TTFP p50 | TTFP p95 |
|---|---|---:|---:|
| A0 | vLLM target-only + Torch Flow | 318.8 ms | 392.9 ms |
| A1 | A0 + DSpark LLM | 267.4 ms | 325.3 ms |
| A2 | A1 + FlashInfer DiT | 146.0 ms | 159.7 ms |
| A3 | A2 + packed DiT batching | 137.7 ms | 148.7 ms |
| A4 | A3 + DiT CUDA Graph | 121.0 ms | 127.8 ms |
| A5 | A4 + HiFT CUDA Graph | 106.5 ms | 113.3 ms |
| A6 | A5 + CUDA MPS | **72.8 ms** | **82.0 ms** |

![Faster CosyVoice single-concurrency cumulative TTFP ablation: p50 and p95](assets/c1-ttfp-p50-p95.svg)

From A0 to A6, single-concurrency TTFP p50 decreases by 77.2% and p95 by
79.1%.

## Comparison with other frameworks

The following charts compare Faster CosyVoice, the Triton Inference Server
Solution (TRT-LLM + TRT), and vLLM-Omni under the same fixed-concurrency load.
The x-axis is concurrent requests; lower TTFP and higher RTFx are better.

### TTFP p50

![CosyVoice3 fixed-concurrency TTFP p50](assets/cosyvoice3-fixed-concurrency-p50-ttfp.svg)

### TTFP p95

![CosyVoice3 fixed-concurrency TTFP p95](assets/cosyvoice3-fixed-concurrency-p95-ttfp.svg)

### RTFx

![CosyVoice3 fixed-concurrency audio throughput in RTFx](assets/cosyvoice3-fixed-concurrency-audio-xrt.svg)

## Seed-TTS Chinese WER

We use all 2,020 samples in the Seed-TTS Chinese test set to check whether
DSpark speculative decoding affects generation quality. Each sample uses its
paired `prompt_audio + prompt_text → target_text`, and the generated audio is
decoded with a Paraformer model.

| LLM configuration | Samples | Character errors | Character-WER / CER | Exact ASR matches | Generation failures |
|---|---:|---:|---:|---:|---:|
| Target only (no draft) | 2,020 | 490 | 1.1593% | 1,645 (81.44%) | 0 |
| Target + DSpark draft | 2,020 | **481** | **1.1380%** | **1,649 (81.63%)** | 0 |
