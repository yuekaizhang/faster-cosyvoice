# Faster CosyVoice3 C1 TTFP ablation

Final presentation view containing only fixed concurrency C1 TTFP. The two bars for each cumulative configuration are p50 and p95.

Protocol: single NVIDIA H100 80GB; fixed registered voice; Seed-TTS evaluation text sequence; 15 s warm-up and 60 s measurement; no leading-silence trim; uniform-20 streaming chunks. Every measured first chunk was exactly 760 ms, and all C1 requests completed successfully with zero underruns. A0–A6 use MPS OFF; A7 differs from A6 only by MPS ON.

Raw benchmark root: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/faster-cosyvoice/benchmarks/results/faster-cosyvoice-ablation-v2-760-20260902`

Figure: [C1 TTFP p50 and p95](c1-ttfp-p50-p95.png). Machine-readable values: [c1-ttfp.csv](c1-ttfp.csv).

| Step | Cumulative configuration | TTFP p50 (ms) | TTFP p95 (ms) |
|---|---|---:|---:|
| A0 | Baseline vLLM + Torch | 318.8 | 392.9 |
| A1 | + DSpark | 267.4 | 325.3 |
| A2 | + FlashInfer | 146.0 | 159.7 |
| A3 | + Packed batching | 137.7 | 148.7 |
| A4 | + Deadline scheduler | 137.4 | 150.8 |
| A5 | + Flow CUDA Graph | 121.0 | 127.8 |
| A6 | + HiFT CUDA Graph | 106.5 | 113.3 |
| A7 | + CUDA MPS | 72.8 | 82.0 |

A0 → A7 reduces p50 from 318.8 to 72.8 ms (+77.2%) and p95 from 392.9 to 82.0 ms (+79.1%).

TTFP is the Nari benchmark's `first_playable_ms`. No C8 or xRT values are included in this final view.
