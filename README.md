# faster-cosyvoice

CosyVoice3 加速 voice-clone 推理：vLLM + DSpark 投机解码（LLM）+ FlashInfer DiT（token2wav）。
设计文档：docs/superpowers/specs/2026-08-05-faster-cosyvoice-design.md

## 环境（容器内）
    bash scripts/setup_env.sh
    source venv/bin/activate
    export PYTHONPATH=$PWD/third_party/spec-vllm:$PYTHONPATH
    export HF_HOME=...   # 可选，指向已有缓存

环境为分层复用：vllm==0.25.1 wheel（dspark 内置）等重依赖来自 vllm025_venv（.pth 挂载），
详见 requirements.txt 注释与 scripts/setup_env.sh。

## Offline 推理
    # 单条
    python examples/offline_inference.py --ref-audio ref.wav \
        --ref-text "参考文本" --target-text "目标文本" --output-dir results/single
    # 数据集批量（wenetspeech4tts 是 split）
    python examples/offline_inference.py --dataset yuekai/seed_tts_cosy2 \
        --split wenetspeech4tts --batch-size 8 --output-dir results/ws4tts
    # 关投机解码 / torch estimator 降级
    --draft-model none / --estimator torch

脚本内部 `OMP_NUM_THREADS=1`（vLLM EngineCore fork 后 OpenMP 会 segfault），可用环境变量覆盖。

## 测试
    pytest                     # CPU 单测（gpu/integration 默认跳过）
    pytest tests/gpu -m gpu -v # offline e2e + server e2e + interleave 回归（容器内）
    python scripts/asr_check.py --wav-dir results/... --ref-json results/.../expected.json \
        --paraformer-dir models/sherpa-onnx-paraformer-zh-2023-09-14

## 基准（H100，yuekai/seed_tts_zh_cosy3 test_zh 200 条，bs16）

复现命令：

    python examples/offline_inference.py --dataset yuekai/seed_tts_zh_cosy3 \
        --split test_zh --limit 200 --batch-size 16 --output-dir results/bench_dspark
    python examples/offline_inference.py --dataset yuekai/seed_tts_zh_cosy3 \
        --split test_zh --limit 200 --batch-size 16 --draft-model none \
        --output-dir results/bench_baseline

| 配置 | llm tok/s | 平均接受长度 | 端到端 RTF |
|---|---|---|---|
| dspark | 9387.6 | 2.992 | 0.0110 |
| baseline（无 draft） | 2057.5 | — | 0.0203 |

llm 加速比 4.56x。注意：SpeechSpec 参考值为 bs16 下 1.70x（全量 2020 条）。本仓数字更高的可能原因：
仅 200 条（continuous batching 尾部效应不同）、0.5B 模型 bs16 下 per-step 开销占主导使步数减少收益
放大、计时口径差异（本仓只计 llm.generate 墙钟）。接受长度 2.992 与参考值 2.98 一致，说明投机
解码行为本身对齐。
RTF 为 (llm+t2w)/音频时长，不含 frontend。

## Streaming server（M2）

    bash scripts/run_server.sh --port 8000          # 启动（含引擎加载+warmup，~2分钟）
    # 流式请求（客户端示例，打印 TTFA/时长）：
    python examples/stream_client.py --url http://localhost:8000 \
        --ref-audio ref.wav --ref-text "参考文本" --target-text "目标文本" --out out.wav
    # curl（非流式）：
    curl -X POST http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
        -d '{"input":"你好","voice":"my_voice"}' -o out.wav
    # 音色注册 / 列表：
    curl -X POST http://localhost:8000/v1/audio/voices -H 'Content-Type: application/json' \
        -d '{"name":"my_voice","ref_audio":"data:audio/wav;base64,...","ref_text":"参考文本"}'
    curl http://localhost:8000/v1/audio/voices

协议：OpenAI `/v1/audio/speech`（扩展 `ref_audio`/`ref_text`/`seed`），`response_format` wav|pcm，
`stream:true` 为增量 PCM（首包 44 字节未知长度 WAV 头）。v1 流式 token2wav 用 torch estimator
（chunk-causal + 确定性重算；flashinfer 流式是 M3）；LLM 侧 DSpark 投机解码照常生效。
音色注册为内存态，重启即失。`--draft-model none` 关投机解码；`--gpu-memory-utilization` 默认 0.5。

实测（H100，4 并发流式，e2e 测试输出）：server 侧 TTFA 稳态 286–494ms（并发首请求含
frontend 处理约 0.5–1.1s）；ASR 门（CER≤0.15）流式/非流式 5/5 通过，同 seed 流式与非流式
转写完全一致。交错回归测试锁定多 session bit-exact（tests/gpu/test_interleave.py）。

M1 offline + M2 streaming server 已落地；M3（flashinfer 流式 mask + 跨请求 packed batch）见设计文档里程碑。
