# Faster CosyVoice Benchmark

本文档记录 Faster CosyVoice 的性能指标、累积消融结果，以及与其他 CosyVoice3
推理框架的性能对比。

## 指标与测试方法

| 指标 | 定义 | 解读 |
|---|---|---|
| TTFP | 从发起请求到客户端收到第一块可播放 PCM 的时间 | 越低越好 |
| p50 TTFP | 所有请求 TTFP 的中位数 | 表示典型请求延迟 |
| p95 TTFP | 95% 请求不超过的 TTFP | 表示尾延迟 |
| RTFx | 测量窗口内收到的音频总时长 / 测量窗口时长 | 越高越好；`1 RTFx` 表示实时生成 |

本文统一使用 **RTFx** 表示音频吞吐。它与单请求常用的 RTF 方向相反：RTF 是计算
时间除以音频时长，越低越好；RTFx 是单位墙钟时间生成的音频时长，越高越好。

固定并发 `C` 表示同时运行 `C` 个 worker；每个 worker 必须等待当前请求完整结束，
才会发送下一条请求。因此，横轴表示持续存在的并发请求数。
测试协议如下：

- 单张 NVIDIA Hopper GPU；
- 固定参考音频和参考文本；
- 15 秒 warmup，随后测量 60 秒；
- seed 为 `0`；
- leading-silence trim 关闭。

## Faster CosyVoice 累积消融

消融测试统一使用单并发。所有请求均完整返回、没有 underrun，
首块音频长度均为 760 ms。A0–A5 关闭 MPS，A6 只在 A5 基础上开启 MPS；每一行
都包含此前步骤的全部优化。

| Step | 累积配置 | TTFP p50 | TTFP p95 |
|---|---|---:|---:|
| A0 | vLLM target-only + Torch Flow | 318.8 ms | 392.9 ms |
| A1 | A0 + DSpark LLM | 267.4 ms | 325.3 ms |
| A2 | A1 + FlashInfer DiT | 146.0 ms | 159.7 ms |
| A3 | A2 + packed DiT batching | 137.7 ms | 148.7 ms |
| A4 | A3 + DiT CUDA Graph | 121.0 ms | 127.8 ms |
| A5 | A4 + HiFT CUDA Graph | 106.5 ms | 113.3 ms |
| A6 | A5 + CUDA MPS | **72.8 ms** | **82.0 ms** |

![Faster CosyVoice 单并发 TTFP 累积消融：p50 与 p95](assets/c1-ttfp-p50-p95.svg)

从 A0 到 A6，单并发 TTFP p50 降低 77.2%，p95 降低 79.1%。

## 与其他框架对比

下面在相同的固定并发负载下比较 Faster CosyVoice、Triton Inference Server Solution
(TRT-LLM + TRT) 和 vLLM-Omni。横轴为并发请求数，TTFP 越低越好，RTFx 越高越好。

### TTFP p50

![CosyVoice3 固定并发 TTFP p50](assets/cosyvoice3-fixed-concurrency-p50-ttfp.svg)

### TTFP p95

![CosyVoice3 固定并发 TTFP p95](assets/cosyvoice3-fixed-concurrency-p95-ttfp.svg)

### RTFx

![CosyVoice3 固定并发音频吞吐 RTFx](assets/cosyvoice3-fixed-concurrency-audio-xrt.svg)

## Seed-TTS 中文 WER

使用 Seed-TTS 中文测试集的 2,020 条样本检查 DSpark speculative decoding 是否影响
生成质量。每条样本均使用各自配套的 `prompt_audio + prompt_text → target_text`，
生成结果由 Paraformer 模型解码。

| LLM 配置 | 样本数 | 字符错误数 | Character-WER / CER | ASR 完全匹配 | 生成失败 |
|---|---:|---:|---:|---:|---:|
| Target only（无 draft） | 2,020 | 490 | 1.1593% | 1,645（81.44%） | 0 |
| Target + DSpark draft | 2,020 | **481** | **1.1380%** | **1,649（81.63%）** | 0 |
