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
    pytest tests/gpu -m gpu -v # GPU e2e（容器内）
    python scripts/asr_check.py --wav-dir results/... --ref-json results/.../expected.json \
        --paraformer-dir models/sherpa-onnx-paraformer-zh-2023-09-14

## 基准（H100，yuekai/seed_tts_zh_cosy3 test_zh 200 条，bs16）
| 配置 | llm tok/s | 平均接受长度 | 端到端 RTF |
|---|---|---|---|
| dspark | 9387.6 | 2.992 | 0.0110 |
| baseline（无 draft） | 2057.5 | — | 0.0203 |

llm 加速比 4.56x（0.5B 模型 bs16 下 per-step 开销占主导，投机解码减少步数收益超过接受长度本身）。
RTF 为 (llm+t2w)/音频时长，不含 frontend。

M1 = offline；M2 streaming server / M3 flashinfer 流式见设计文档里程碑。
