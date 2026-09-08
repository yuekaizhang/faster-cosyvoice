# CosyVoice3 fixed-concurrency benchmark（760 ms 对齐后）

这组结果使用单卡 NVIDIA H100 80GB、Triton 2.59.0、TensorRT-LLM
0.20.0 和 10 个 CosyVoice3 BLS instance。客户端采用 Nari 指标协议的 closed-loop
固定并发：每个 worker 必须等上一条响应完整结束后才发送下一条请求；每个点预热 15 秒、
测量 60 秒、seed 0。请求使用固定 reference voice，target text 来自同一份
`seed-tts-eval.jsonl` 确定性序列。

## Padding 修复

Triton BLS 过去忽略了 prompt speech token 在整个 token timeline 中的位置，首轮固定消费
15 个 target token，对本次 97-token prompt 只产生 440 ms 首块。现在首轮按下面的协议消费
真实生成 token（不插入伪 token）：

```text
prompt_token_pad = (-prompt_token_len) % token_hop_len
first_target_hop = token_hop_len + prompt_token_pad
```

因此本次 `prompt_token_pad=8`，首块与 faster-cosyvoice 和 vLLM-Omni 一样为
760 ms。20 个正式 run 的所有 measurement 请求均验证为精确 760 ms。

## Triton + TRT-LLM：MPS OFF

| C | RPS | TTFP p50 (ms) | TTFP p95 (ms) | audio xRT | underrun | PCM / audible / success |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.933 | 197.1 | 204.8 | 8.648 | 0/116 (0.0%) | 116 / 116 / 116 |
| 2 | 2.467 | 293.2 | 339.1 | 11.472 | 0/148 (0.0%) | 148 / 147 / 148 |
| 3 | 2.767 | 396.3 | 485.3 | 12.848 | 0/166 (0.0%) | 166 / 165 / 166 |
| 4 | 2.883 | 494.8 | 565.9 | 12.914 | 0/173 (0.0%) | 173 / 173 / 173 |
| 6 | 2.850 | 709.0 | 773.3 | 12.955 | 0/171 (0.0%) | 171 / 171 / 171 |
| 8 | 2.867 | 898.9 | 997.4 | 13.011 | 62/172 (36.0%) | 172 / 172 / 172 |
| 10 | 2.883 | 1091.0 | 1188.7 | 13.023 | 150/173 (86.7%) | 173 / 173 / 173 |
| 12 | 2.917 | 1255.1 | 1396.7 | 13.215 | 173/175 (98.9%) | 175 / 175 / 173 |
| 14 | 3.000 | 1461.2 | 1587.0 | 13.323 | 179/180 (99.4%) | 180 / 180 / 84 |
| 16 | 3.050 | 1643.8 | 1785.9 | 13.357 | 183/183 (100%) | 183 / 183 / 33 |

## Triton + TRT-LLM：MPS ON

| C | RPS | TTFP p50 (ms) | TTFP p95 (ms) | audio xRT | underrun | PCM / audible / success |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.567 | 130.3 | 136.0 | 11.360 | 0/154 (0.0%) | 154 / 154 / 154 |
| 2 | 4.283 | 148.3 | 193.1 | 19.383 | 0/257 (0.0%) | 257 / 256 / 257 |
| 3 | 5.250 | 184.6 | 237.8 | 25.147 | 0/315 (0.0%) | 315 / 314 / 315 |
| 4 | 5.783 | 237.6 | 295.6 | 27.170 | 0/347 (0.0%) | 347 / 346 / 347 |
| 6 | 5.983 | 334.9 | 397.1 | 27.473 | 0/359 (0.0%) | 359 / 358 / 359 |
| 8 | 5.983 | 430.9 | 496.5 | 27.281 | 0/359 (0.0%) | 359 / 358 / 359 |
| 10 | 6.017 | 518.1 | 581.8 | 27.397 | 0/361 (0.0%) | 361 / 361 / 361 |
| 12 | 6.100 | 611.3 | 680.6 | 27.222 | 0/366 (0.0%) | 366 / 366 / 366 |
| 14 | 5.917 | 711.1 | 810.5 | 27.524 | 5/355 (1.4%) | 355 / 355 / 355 |
| 16 | 5.917 | 792.2 | 896.9 | 27.637 | 56/355 (15.8%) | 355 / 354 / 355 |

`success` 还应用 continuity gate（单次 stall 不超过 500 ms、累计 stall 不超过
1000 ms）；所以完整 PCM 仍可能因播放停顿而不计 success。`audible` 的个别差一条是
动态可听起点检测结果，不代表 PCM 不完整。

## 结论

- MPS OFF 的吞吐平台约为 3.0 RPS / 13.4x realtime；MPS ON 约为
  6.1 RPS / 27.6x realtime，接近翻倍。
- MPS ON 的连续播放甜点区是 C10–C12：C12 仍为 0 underrun，C14 开始出现
  1.4%，C16 上升到 15.8%。MPS OFF 则从 C8 开始明显 underrun。
- Padding 修复主要增加首块播放 buffer，不会凭空提高模型吞吐。MPS OFF C6 从旧版
  440 ms 的 153/174 underrun 降为 760 ms 的 0/171，而 RPS 仍约 2.9；MPS ON
  C10 从 129/355 降为 0/361，RPS 仍约 6.0。

## 文件

- `cosyvoice3-fixed-concurrency-p50-ttfp.{png,svg}`：p50 首包延迟
- `cosyvoice3-fixed-concurrency-p95-ttfp.{png,svg}`：p95 首包延迟
- `cosyvoice3-fixed-concurrency-actual-rps.{png,svg}`：请求吞吐
- `cosyvoice3-fixed-concurrency-audio-xrt.{png,svg}`：生成音频 xRT
- `cosyvoice3-fixed-concurrency-underrun-rate.{png,svg}`：模拟播放 underrun 比例
- `cosyvoice3-fixed-concurrency-first-chunk.{png,svg}`：首块音频时长
- `cosyvoice3-fixed-concurrency-p95-ttfp-vs-rps.{png,svg}`：延迟/吞吐 frontier
- `triton-padding-before-after-*.{png,svg}`：Triton 440 ms 与 760 ms 的修复前后对照
- `cosyvoice3-fixed-concurrency-data.csv`：当前所有后端绘图数据
- `triton-padding-before-after-data.csv`：padding 修复前后绘图数据

原始 Triton 结果目录前缀：

```text
benchmarks/results/fixed-triton-trtllm-padding760-mps-off-concurrency-*
benchmarks/results/fixed-triton-trtllm-padding760-mps-on-concurrency-*
```

重绘命令：

```bash
python3 scripts/plot_fixed_concurrency_benchmark.py
```
