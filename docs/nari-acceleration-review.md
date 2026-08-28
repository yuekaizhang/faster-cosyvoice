# Nari Qwen3-TTS 加速方案对照

本文对照以下两份 2026-08-28 的本地快照与 Nari 的公开说明：

- `../nari/nari-qwen3-tts`：`e8c5b2b`
- `../nari/benchmarks`：`a2d65e1`
- [Nari: Realtime TTS at the Speed-of-Light](https://nari-labs.com/blog/qwen3-tts-speed-cost-frontier/)

结论先行：最值得移植的不是某一个 kernel，而是 **增量 Codec 状态缓存**。
当前 faster-cosyvoice 已经覆盖了 Nari 的一部分调度、CUDA Graph 和自回归生成优化，
但 Flow 和 HiFT 仍会在每个流式 chunk 上重算历史前缀；这会随输出增长，成为持续吞吐和
高负载 TTFA 的主要瓶颈。

## 可迁移性和优先级

| Nari 手段 | 本仓现状 | 可迁移性 | 建议 |
|---|---|---:|---|
| Codec Transformer/CNN 增量状态缓存 | Flow 每块重算完整 token 前缀；HiFT 每块重算累计 mel | 高 | **P0**：先缓存 HiFT/Flow 可复用状态，这是最大的结构性收益 |
| 首音频与持续播放 deadline-aware 调度 | **已实现**：route-boundary playback credit + urgent/startup/EDF | 高 | 默认启用；Nari open-loop RPS=1 已消除旧调度的 chunk-wave 饥饿 |
| 固定 batch-size CUDA Graph | 单 session Flow 和 HiFT 已有长度桶；packed `B>1` 仍走 eager | 高 | **P1**：为常见 packed batch size 捕图，过大 cohort 拆分 |
| 小首块、后续增大 | 默认 `15, 30, 60...`；低 TTFA 配置是 uniform-25 | 高 | **P1**：用 Nari benchmark 联合搜索首块/后续块，目标同时约束 TTFA 与 underrun |
| 动态裁剪 leading silence | **已实现（opt-in）**：同一 audible detector、20ms pre-roll、bounded fallback、startup buffer | 高 | `--trim-leading-silence`；改善 audible TTFA，不改善计算本身 |
| 把固定 Code Predictor 循环捕成单一 CUDA Graph | CosyVoice 没有 Qwen3-TTS 的同构 15-step Predictor | 低 | 不直接移植；本仓的对应优化是 vLLM DSpark speculative decoding |
| 短上下文定制 attention kernel | Flow 已有 FlashInfer chunk-causal 路径 | 中 | 对 profile 结果中占比最高的短 shape 再做 Triton/融合 kernel |
| 避免 CPU/GPU 同步与逐步 EOS 检查 | vLLM 与 token2wav 路径仍有可审计同步点 | 中 | **P2**：用 profiler 定位，不做无证据的全局异步改写 |
| 统一 Talker/Predictor/Codec scheduler | LLM EngineCore 与 token2wav 是独立进程/调度器 | 低/重构大 | 长期项；短期用 MPS + 跨 session packed batching 获取大部分并发收益 |
| FP8 | 当前 BF16/FP16；H100 支持 FP8 | 中 | **P2**：先做 LLM/Flow 分组件精度与 ASR 门，再决定默认值 |
| 输入文本流式传输 | HTTP 当前接收完整文本 | 中 | 语音对话场景有价值；对 Nari 当前完整文本 benchmark 没有收益 |

## 已经具备的对应能力

1. **自回归阶段**：vLLM 0.25.1 + DSpark，已有离线实测 `9387.6 tok/s`，
   相比无 draft 的 `2057.5 tok/s` 为 `4.56x`。
2. **播放 deadline 调度**：`faster_cosyvoice/streaming/batcher.py` 以首次 PCM
   route 时间与累计 PCM 时长计算 deadline；urgent established stream 优先，
   其余情况下 startup 优先，再按 EDF，避免旧 `(chunk_index, arrival)` 的 wave。
3. **跨请求组批**：packed 模式把不同 session 的 ready chunk 合并到一次
   FlashInfer Flow forward。
4. **CUDA Graph**：Flow 和 HiFT 都支持长度 bucket；低 TTFA 配置用固定的
   uniform-25 chunk 让 shape 可枚举。
5. **同卡并发**：CUDA MPS 允许独立的 vLLM EngineCore 与 token2wav 进程并发执行。
   仓库此前的内部“首个 PCM chunk”中位数从 `104.6 ms` 降至 `76.1 ms`。
6. **首段静音抑制**：可选服务端 gate 与 benchmark 使用完全相同的 Nari v1 onset
   规则；确认 onset 后保留 20 ms pre-roll，并累计 startup buffer 后才开始播放。

## 指标口径

仓库旧结果里的 TTFP 是服务内部或客户端收到首个 PCM chunk 的时间。Nari 的主要指标是
**audible TTFA**：从请求发出到模拟即时播放首次达到可听阈值的时间。它还分别记录：

- TTFB：收到首个非空 HTTP body byte；
- first playable：收到第一个完整 PCM frame；
- leading silence：音频开头到持续可听声的时长；
- underrun：按音频到达时间模拟零预缓冲播放后的断流次数。

本仓的 `examples/stream_client.py` 和 `scripts/run_nari_benchmark.sh` 使用 Nari v1
检测规则：24 kHz mono PCM16、20 ms RMS 窗、10 ms hop、-45 dBFS、连续两个活跃窗。
因此新的 audible TTFA 不应与旧的 76.1 ms 内部 TTFP 直接横向比较。

## Benchmark 公平性

- 使用 Nari 固定的 1,088 条英文 Seed-TTS prompt projection；
- 使用 Poisson open-loop arrival，而不是“上一个完成后才发下一个”的 closed loop；
- 每个 rate point 独立运行，30 秒 warmup、5 分钟 measurement；
- 先注册一个固定 voice，使每请求不包含参考音频下载和 CampPlus 前处理；
- 原始 PCM 流式返回，客户端模拟立即播放；
- 没有配置 Deepgram key 时不报告 WER，只报告延迟、连续性、错误率与音频时长。

这复用了 Nari 的方法和客户端实现，但不是模型间的同质量对比：Nari 测的是
Qwen3-TTS CustomVoice，本仓测的是 CosyVoice3 zero-shot voice clone。

## 本机 5 分钟结果

测试环境为单张 H100 80GB、CUDA MPS、DSpark、packed FlashInfer、uniform-25、
Flow/HiFT graph buckets、deadline-aware 调度和 bounded leading-silence trim；固定
benchmark voice，seed 0。每个 rate point 独立运行 30 秒 warmup + 5 分钟
measurement。raw PCM 下 TTFB 与 first playable 相同，是本仓最接近“首 PCM TTFP”
的客户端指标。

| 请求 RPS（实际） | 请求数 | TTFB p50 / p95 / p99 | audible TTFA p50 / p95 / p99 | E2E p95 | 完整 PCM | underrun |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (1.073) | 322 | 84.2 / 186.7 / 214.8 ms | 104.2 / 206.7 / 234.8 ms | 444.7 ms | 322/322 | 0 |
| 6 (5.853) | 1,756 | 218.1 / 422.3 / 558.7 ms | 238.1 / 442.5 / 582.8 ms | 1551.9 ms | 1,756/1,756 | 0 |

RPS=1 的 322 条都达到 audible threshold。RPS=6 只有 1,755/1,756 条可听：异常
请求完整返回 11.2 秒音频，但最高 20ms-frame RMS 仅 -53.58 dBFS，低于 Nari 的
-45 dBFS threshold。用相同文本和 seed 42 单请求重放时，输出在 20ms 达到可听，
所以目前不能把异常归因于文本或静音裁剪；并发下的稀有模型/服务质量问题仍需继续定位。
因此 RPS=6 的 transport/capacity 指标通过，Nari semantic-quality gate 为 fail。
本轮没有 Deepgram key，WER 未检查。

## 本机 scout 结果与调度问题

测试环境为单张 H100 80GB、CUDA MPS、DSpark、packed FlashInfer、uniform-25、
Flow/HiFT graph buckets；固定 benchmark voice，seed 0，15 秒 warmup + 60 秒
measurement。它是找 frontier 的 scout，不替代提交结果所需的 30 秒 + 5 分钟运行。

旧调度在请求 RPS=1（seed 对应实际 1.117 RPS）就出现 chunk wave：TTFB p95
`28.119s`、audible TTFA p95 `28.859s`、40.30% 请求 underrun、peak in-flight 41。
换成 deadline-aware 后，相同 arrival sequence 的 TTFB p95 为 `201.481ms`，
audible TTFA p95 `905.592ms`，67/67 成功、0 underrun、peak in-flight 4。

未裁剪前导静音时的 frontier：

| 请求 RPS（实际） | 请求数 | TTFB p50 / p95 | audible TTFA p50 / p95 | 成功率 | underrun 请求 | peak in-flight |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (1.117) | 67 | 80.8 / 201.5 ms | 341.3 / 905.6 ms | 100% | 0% | 4 |
| 2 (2.267) | 136 | 91.1 / 224.0 ms | 349.5 / 921.1 ms | 100% | 0% | 5 |
| 4 (4.267) | 256 | 151.4 / 298.9 ms | 390.7 / 976.3 ms | 100% | 0% | 9 |
| 6 (6.350) | 381 | 241.2 / 416.3 ms | 469.9 / 1023.0 ms | 100% | 0% | 17 |
| 8 (8.200) | 492 | 334.7 / 452.1 ms | 570.2 / 1123.5 ms | 100% | 0% | 43 |
| 10 (9.817) | 589 | 7738.9 / 12735.5 ms | 7985.9 / 13120.0 ms | 77.42% | 94.91% | 148 |

因此该配置在这批文本上的容量边界位于 8–10 RPS。RPS=8 虽然零 underrun，
但 E2E p95 已到 5.19 秒；RPS=10 明确过载。audible TTFA 远大于 TTFB 的主要
原因是模型生成的 leading silence（例如 RPS=1 p95 830 ms），所以 silent-first
优化必须和计算首包延迟分开报告。

打开 `--trim-leading-silence` 并在首次 route 前确保 400 ms startup buffer 后：

| 请求 RPS（实际） | TTFB p50 / p95 | audible TTFA p50 / p95 | leading silence p95 | 成功率 | underrun |
|---:|---:|---:|---:|---:|---:|
| 1 (1.117) | 85.0 / 202.4 ms | 105.0 / 222.4 ms | 20 ms | 100% | 0 |
| 6 (6.350) | 276.2 / 520.3 ms | 296.2 / 540.3 ms | 20 ms | 100% | 0 |

400 ms buffer 仅在裁剪后首个 payload 太短时延后 route；它不把所有请求固定延迟
400 ms。RPS=6 的 TTFB p95 比不裁剪增加约 104 ms，但 audible TTFA p95 减少约
483 ms，并保持严格零 underrun。未运行 Deepgram WER，因此这里只能确认被删除的
字节位于 Nari 阈值定义的前导静音区，不能把本轮结果当作完整语义质量结论。
