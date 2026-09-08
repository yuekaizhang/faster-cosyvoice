# Triton 容器内复测环境恢复手册

这份手册记录 2026-09-01 实际跑通的单卡 H100、固定并发复测环境。容器根文件系统
会随时限消失；代码、模型缓存、uv cache、benchmark 数据和结果都保存在 Lustre。

## 1. 路径和版本

```bash
export TTS_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts
export FCV_ROOT=$TTS_ROOT/faster-cosyvoice
export COSYVOICE_ROOT=$TTS_ROOT/CosyVoice
export SGLANG_ROOT=$TTS_ROOT/sglang-omni
export NARI_ROOT=$TTS_ROOT/nari
cd "$FCV_ROOT"
```

本次有效版本：

| 组件 | Revision/version |
|---|---|
| faster-cosyvoice | `ed7cb7d290526fd4a6770722198f9efaf353a48a` |
| CosyVoice Triton checkout | `ee734e0d24f371e107ed37bec112997e9557475d` |
| SGLang-Omni | `7bbdac6eef10ebc308cee36dd1f9b2b1b68084cb` |
| SGLang | `0.5.18` |
| Triton Server | `2.59.0` |
| TensorRT-LLM | `0.20.0` |
| TensorRT | `10.10.0.31` |
| PyTorch（SGLang） | `2.13.0+cu130` |
| uv | `0.11.28` |

## 2. 每个新容器先做一次

Triton 镜像已有 CUDA/Python 主体，但需要驱动兼容库和 TorchCodec 能加载的 FFmpeg 6
动态库。这里安装的是库，不依赖 `ffmpeg` 命令行程序。

```bash
apt-get update
apt-get install -y \
  cuda-compat-13-0 \
  libavutil58 libavcodec60 libavformat60 libavdevice60 \
  libavfilter9 libswscale7 libswresample4

python3 -m pip install uv==0.11.28
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
```

三个持久化 venv 的 Python symlink 指向容器内的 `/root/.local/share/uv/python`。新容器
先恢复同一 Python 3.12.13 安装，通常不需要重装 243 个项目依赖：

```bash
export UV_CACHE_DIR=$FCV_ROOT/.cache/uv
uv python install 3.12.13

$FCV_ROOT/.venv/bin/python -V
$FCV_ROOT/.benchmark-venv/bin/python -V
$SGLANG_ROOT/.venv/bin/python -V
```

三条都成功即可跳过同步。如果 faster-cosyvoice venv 不完整，再执行：

```bash
cd "$FCV_ROOT"
source scripts/env.sh
UV_LINK_MODE=hardlink uv sync --frozen
```

如果 Nari benchmark venv 不完整，再执行：

```bash
cd "$FCV_ROOT"
NARI_BENCH_DIR=$NARI_ROOT/benchmarks/tts_bench \
BENCH_VENV=$FCV_ROOT/.benchmark-venv \
bash scripts/setup_nari_benchmark.sh
```

不要手工创建 `cpython-3.12-linux-x86_64-gnu` 目录后再运行 uv；uv 会把该名字作为
minor-version alias 使用，普通目录会与 alias 冲突。先运行 `uv python install` 最稳。

## 3. 固定 reference 和 benchmark client

所有后端使用同一条 reference：

```text
WAV:  benchmarks/data/benchmark-ref-test-en-0.wav
SHA:  af3ac928e98cdb171c9abdc611813f2cc40c52b71167790b785628104f81f389
Text: We asked over twenty different people, and they all said it was his.
Data: yuekai/seed_tts_cosy2:test_en[0]
```

数据集 `benchmarks/data/seed-tts-eval.jsonl` 的 SHA-256 是
`c95cb482f71117cbc46ac4e3aa5eab5c199bb0386d9e5600d912e157da8d2866`。

SGLang 的 local-media 安全检查会解析真实路径，不能把 Lustre WAV 软链接到 `/tmp`；
必须复制成普通文件：

```bash
cp "$FCV_ROOT/benchmarks/data/benchmark-ref-test-en-0.wav" /tmp/benchmark-ref.wav
sha256sum /tmp/benchmark-ref.wav
```

固定并发 client 是 closed-loop：`C` 个 worker 各自只保留一个在途请求，收到完整响应
后才发下一条。标准 sweep：

```bash
cd "$FCV_ROOT"
export BENCH_VENV=$FCV_ROOT/.benchmark-venv
export CONCURRENCIES='1 2 3 4 6 8 10'
export WARMUP=15s
export DURATION=60s
export STOP_ON_INVALID_AUDIO=0
export OUTPUT_PREFIX=$FCV_ROOT/benchmarks/results/retest-backend-mps-mode-20260901-01
bash scripts/run_fixed_concurrency_sweep.sh
```

结果目录不可复用；每次必须换新的 `OUTPUT_PREFIX`。所有响应 WAV 自动保存在每个结果
目录的 `audio/wav/{warmup,measurement}/`。

## 4. CUDA MPS ON / OFF

MPS ON 时必须在启动任何 CUDA client 之前启动 daemon，并让所有相关进程继承完全相同
的 pipe/log 变量：

```bash
export MPS_RUN_ROOT=/tmp/mps-cosyvoice-retest
mkdir -p "$MPS_RUN_ROOT/pipe" "$MPS_RUN_ROOT/log"
export CUDA_MPS_PIPE_DIRECTORY=$MPS_RUN_ROOT/pipe
export CUDA_MPS_LOG_DIRECTORY=$MPS_RUN_ROOT/log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d
printf 'set_default_active_thread_percentage 100\n' | nvidia-cuda-mps-control
printf 'get_default_active_thread_percentage\nget_server_list\n' | nvidia-cuda-mps-control
```

拿到 `get_server_list` 输出的 server PID 后验证真实 client attachment：

```bash
MPS_SERVER_PID=123456  # 替换为上一条命令返回的 PID
printf "get_client_list $MPS_SERVER_PID\n" | nvidia-cuda-mps-control
```

仅设置环境变量不等于已接入 MPS。faster-cosyvoice 应看到两个主要 CUDA client；
Triton + TRT-LLM 应看到 TRT-LLM 和 token2wav 相关 client；当前 SGLang CosyVoice3
只有一个 GPU client。

MPS OFF 启动前清掉变量：

```bash
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
```

测试结束并先停止服务后关闭 daemon：

```bash
printf 'quit\n' | nvidia-cuda-mps-control
```

## 5. Triton + TensorRT-LLM

已经生成的 BF16 engine 和 `model_repo_cosyvoice3_copy` 都在 Lustre，无需每个容器重新
构建。先设置：

```bash
export TRITON_CV_ROOT=$COSYVOICE_ROOT/runtime/triton_trtllm
export PYTHONPATH=$COSYVOICE_ROOT:$COSYVOICE_ROOT/third_party/Matcha-TTS:${PYTHONPATH:-}
export HF_HOME=$FCV_ROOT/.cache/huggingface
```

在三个终端启动 TRT-LLM、Triton Server 和无调度/无额外 buffer 的 HTTP bridge。
MPS ON 时三个终端都必须保留第 4 节的 MPS 环境变量；OFF 时三个终端都必须 unset。

终端 A：

```bash
CUDA_VISIBLE_DEVICES=0 \
mpirun -np 1 --allow-run-as-root --oversubscribe \
trtllm-serve serve \
  --tokenizer "$TRITON_CV_ROOT/hf_cosyvoice3_llm" \
  "$TRITON_CV_ROOT/trt_engines_bfloat16" \
  --max_batch_size 64 \
  --kv_cache_free_gpu_memory_fraction 0.4
```

`mpirun -np 1 --allow-run-as-root --oversubscribe` 不能省。直接运行 `trtllm-serve`
可能让 health 表面 ready，但第一条真实请求触发 `MPI_ERR_SPAWN`。

终端 B：

```bash
CUDA_VISIBLE_DEVICES=0 tritonserver \
  --model-repository "$TRITON_CV_ROOT/model_repo_cosyvoice3_copy" \
  --http-port 18000 \
  --grpc-port 18001 \
  --metrics-port 18002
```

终端 C：

```bash
cd "$FCV_ROOT"
python3 scripts/triton_nari_bridge.py \
  --host 127.0.0.1 \
  --port 19000 \
  --triton-url 127.0.0.1:18001 \
  --model-name cosyvoice3 \
  --reference-wav benchmarks/data/benchmark-ref-test-en-0.wav \
  --reference-text 'We asked over twenty different people, and they all said it was his.'
```

检查真正 ready：

```bash
curl -fsS http://127.0.0.1:18000/v2/health/ready
curl -fsS http://127.0.0.1:19000/health
```

固定并发参数：

```bash
cd "$FCV_ROOT"
TARGET=nari BASE_URL=http://127.0.0.1:19000 \
MODEL=cosyvoice3 VOICE=benchmark \
CONCURRENCIES='1 2 3 4 6 8 10' \
WARMUP=15s DURATION=60s STOP_ON_INVALID_AUDIO=0 \
OUTPUT_PREFIX=$FCV_ROOT/benchmarks/results/retest-triton-mps-mode-20260901-01 \
bash scripts/run_fixed_concurrency_sweep.sh
```

## 6. faster-cosyvoice

启动时直接调用 venv entry point，避免每次 `uv run` 在 Lustre 上重新扫描；若第 2 节
检查失败才需要 `uv sync`。

MPS OFF：

```bash
cd "$FCV_ROOT"
source scripts/env.sh
env -u CUDA_MPS_PIPE_DIRECTORY \
    -u CUDA_MPS_LOG_DIRECTORY \
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-} \
    "$FCV_VENV/bin/faster-cosyvoice-server" --host 127.0.0.1 --port 8000
```

MPS ON：先执行第 4 节，再在同一环境中启动：

```bash
cd "$FCV_ROOT"
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-} \
"$FCV_VENV/bin/faster-cosyvoice-server" --host 127.0.0.1 --port 8000
```

每次服务重启后都要重新注册 voice；voice 只存在进程内存中：

```bash
cd "$FCV_ROOT"
"$FCV_VENV/bin/python" examples/register_voice.py \
  --url http://127.0.0.1:8000 \
  --name benchmark \
  --ref-audio benchmarks/data/benchmark-ref-test-en-0.wav \
  --ref-text 'We asked over twenty different people, and they all said it was his.'
curl -fsS http://127.0.0.1:8000/health
```

固定并发参数：

```bash
TARGET=nari BASE_URL=http://127.0.0.1:8000 \
MODEL=yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF VOICE=benchmark \
CONCURRENCIES='1 2 3 4 6 8 10 12 14 16' \
WARMUP=15s DURATION=60s STOP_ON_INVALID_AUDIO=0 \
OUTPUT_PREFIX=$FCV_ROOT/benchmarks/results/retest-faster-mps-mode-20260901-01 \
bash scripts/run_fixed_concurrency_sweep.sh
```

启动日志应确认 `Qwen3DSparkModel`、`num_spec_tokens=7` 和 asynchronous scheduling，
否则不是本次被测配置。

## 7. SGLang-Omni

SGLang 使用精确 CosyVoice/Matcha revision。`/tmp` 每个容器都是新的，可从 Lustre
checkout 本地 clone，避免改动用户的 `CosyVoice` worktree：

```bash
export SGLANG_COSYVOICE=/tmp/cosyvoice-sglang-074ca6
git clone --no-hardlinks "$COSYVOICE_ROOT" "$SGLANG_COSYVOICE"
git -C "$SGLANG_COSYVOICE" checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C "$SGLANG_COSYVOICE" submodule update --init --recursive
git -C "$SGLANG_COSYVOICE/third_party/Matcha-TTS" checkout dd9105b34bf2be2230f4aa1e4769fb586a3c824e
```

本次固定使用本地 WeText FST snapshot，避免首次并发下载损坏缓存：

```bash
export WETEXT_MODEL_DIR=$FCV_ROOT/.cache/wetext-fst-fcc767f
cat "$WETEXT_MODEL_DIR/REVISION"
sha256sum "$WETEXT_MODEL_DIR"/{en,zh}/tn/{tagger,verbalizer}.fst
```

revision 应为 `fcc767f`；四个 SHA-256 依次为：

```text
ce8b28103ccf2d936d057a5cd5621ec29ccfedb5f737e48ac8096e7d6af7b2ad
16a6a2ce28b60975e501e965396edbf94a591216a644abd27c8607a2a9aa24cd
4b8b6504c7effcaa067e2fd14ecac3ae2edf9130d229e11abd0ff9b85cbe0390
9fcadad76cafddb2e96b92d892d18249f0ba5fbe60e4d4fc51f466cb6bc7ade9
```

启动：

```bash
cp "$FCV_ROOT/benchmarks/data/benchmark-ref-test-en-0.wav" /tmp/benchmark-ref.wav
export PYTHONPATH=$FCV_ROOT/scripts/sglang_runtime_hook:$SGLANG_COSYVOICE:$SGLANG_COSYVOICE/third_party/Matcha-TTS:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}

cd "$SGLANG_ROOT"
CUDA_VISIBLE_DEVICES=0 .venv/bin/sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --allowed-local-media-path /tmp \
  --host 127.0.0.1 \
  --port 18080
```

MPS ON 时先执行第 4 节并保留变量；OFF 时先 unset。检查：

```bash
curl -fsS http://127.0.0.1:18080/health
```

固定并发参数：

```bash
cd "$FCV_ROOT"
TARGET=sglang-omni BASE_URL=http://127.0.0.1:18080 \
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
CONCURRENCIES='1 2 3 4 6 8 10' \
WARMUP=15s DURATION=60s STOP_ON_INVALID_AUDIO=0 \
OUTPUT_PREFIX=$FCV_ROOT/benchmarks/results/retest-sglang-mps-mode-20260901-01 \
bash scripts/run_fixed_concurrency_sweep.sh \
  --request-param 'ref_audio="file:///tmp/benchmark-ref.wav"' \
  --request-param 'ref_text="We asked over twenty different people, and they all said it was his."'
```

## 8. 完成后的清理和检查

先用 `Ctrl-C` 停服务，再停 MPS。最后应看不到计算进程：

```bash
pgrep -af 'faster-cosyvoice|VLLM::EngineCore|sgl-omni|tritonserver|trtllm-serve|nvidia-cuda-mps' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

正式结果汇总见
[cosyvoice3-h100-nari-benchmark-results-2026-08-28.md](cosyvoice3-h100-nari-benchmark-results-2026-08-28.md)。
