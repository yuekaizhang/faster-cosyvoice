#!/usr/bin/env bash
# Cumulative faster-cosyvoice ablation used by the assets_v2 article figures.
# Every configuration is measured at fixed concurrency C1 and C8.  A0-A6 run
# with CUDA MPS disabled; A7 differs from A6 only by enabling MPS.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$FCV_ROOT"

# NeMo RL containers export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv.  The
# repository environment must never inherit that setting.
unset FCV_VENV
source scripts/env.sh

# The interactive NeMo job may expose all eight allocated GPUs.  This study is
# explicitly single-H100, so ignore an inherited CUDA_VISIBLE_DEVICES unless a
# task-specific override is supplied.
export CUDA_VISIBLE_DEVICES=${ABLATION_CUDA_VISIBLE_DEVICES:-0}
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${ABLATION_LD_LIBRARY_PATH:-/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}

RUN_ROOT=${RUN_ROOT:-$FCV_ROOT/benchmarks/results/faster-cosyvoice-ablation-v2-20260902}
CONFIGS=${CONFIGS:-"A0 A1 A2 A3 A4 A5 A6 A7"}
CONCURRENCIES=${CONCURRENCIES:-"1 8"}
WARMUP=${WARMUP:-15s}
DURATION=${DURATION:-60s}
SEED=${SEED:-0}
PORT=${PORT:-18080}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1200}
CODEC_CHUNK_FRAMES=${CODEC_CHUNK_FRAMES:-20}
BASE_URL=http://127.0.0.1:$PORT
REF_WAV=$FCV_ROOT/benchmarks/data/benchmark-ref-test-en-0.wav
REF_TEXT="We asked over twenty different people, and they all said it was his."

FLOW_BUCKETS=512,640,768,896,1024,1280
HIFT_BUCKETS=64,128,192,256,384,512
MPS_ROOT=${MPS_ROOT:-/tmp/faster-cosyvoice-ablation-v2-mps}
MPS_STARTED=0
SERVER_PID=""

mkdir -p "$RUN_ROOT/logs"

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    sleep 2
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

stop_mps() {
  if [[ "$MPS_STARTED" != 1 ]]; then
    return
  fi
  printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  MPS_STARTED=0
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
}

cleanup() {
  stop_server
  stop_mps
}
trap cleanup EXIT INT TERM

start_mps() {
  # The executable name is longer than Linux's 15-byte comm field, so
  # `pgrep -x` can never match it reliably.  Match the full command instead.
  if pgrep -f '^nvidia-cuda-mps-control -d$' >/dev/null 2>&1; then
    echo "Refusing to reuse an existing CUDA MPS daemon" >&2
    exit 1
  fi
  export CUDA_MPS_PIPE_DIRECTORY=$MPS_ROOT/pipe
  export CUDA_MPS_LOG_DIRECTORY=$MPS_ROOT/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/ready" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited during startup" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server did not become ready within ${READY_TIMEOUT_S}s" >&2
  return 1
}

completed_result() {
  local output=$1
  [[ -f "$output/status.json" ]] &&
    "$FCV_VENV/bin/python" - "$output/status.json" <<'PY'
import json
import sys

raise SystemExit(0 if json.load(open(sys.argv[1]))["state"] == "complete" else 1)
PY
}

for config in $CONFIGS; do
  stop_mps
  common_args=(
    --host 127.0.0.1
    --port "$PORT"
    --token2wav-device cuda:0
    --gpu-memory-utilization 0.5
    --t2w-batch-size 8
    --t2w-deadline-reserve-ms 100
    --codec-chunk-frames "$CODEC_CHUNK_FRAMES"
    --codec-chunk-scale 1
  )
  case "$config" in
    A0)
      label="vLLM target + Torch Flow"
      config_args=(--draft-model none --stream-estimator torch --t2w-batch-mode serial --t2w-scheduler legacy)
      ;;
    A1)
      label="+ DSpark"
      config_args=(--stream-estimator torch --t2w-batch-mode serial --t2w-scheduler legacy)
      ;;
    A2)
      label="+ FlashInfer"
      config_args=(--stream-estimator flashinfer --t2w-batch-mode serial --t2w-scheduler legacy)
      ;;
    A3)
      label="+ packed batching"
      config_args=(--stream-estimator flashinfer --t2w-batch-mode packed --t2w-scheduler legacy)
      ;;
    A4)
      label="+ deadline scheduler"
      config_args=(--stream-estimator flashinfer --t2w-batch-mode packed --t2w-scheduler deadline)
      ;;
    A5)
      label="+ Flow CUDA Graph"
      config_args=(--stream-estimator flashinfer --t2w-batch-mode packed --t2w-scheduler deadline --stream-graph-buckets "$FLOW_BUCKETS")
      ;;
    A6)
      label="+ HiFT CUDA Graph"
      config_args=(--stream-estimator flashinfer --t2w-batch-mode packed --t2w-scheduler deadline --stream-graph-buckets "$FLOW_BUCKETS" --hift-graph-buckets "$HIFT_BUCKETS")
      ;;
    A7)
      label="+ CUDA MPS"
      config_args=(--stream-estimator flashinfer --t2w-batch-mode packed --t2w-scheduler deadline --stream-graph-buckets "$FLOW_BUCKETS" --hift-graph-buckets "$HIFT_BUCKETS")
      start_mps
      ;;
    *)
      echo "Unknown configuration: $config" >&2
      exit 1
      ;;
  esac

  log=$RUN_ROOT/logs/${config}.server.log
  echo "[$config] $label"
  echo "[$config] server args: ${common_args[*]} ${config_args[*]}"
  setsid "$FCV_VENV/bin/faster-cosyvoice-server" \
    "${common_args[@]}" "${config_args[@]}" >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready; then
    tail -n 200 "$log" >&2 || true
    exit 1
  fi

  "$FCV_VENV/bin/python" examples/register_voice.py \
    --url "$BASE_URL" \
    --name benchmark \
    --ref-audio "$REF_WAV" \
    --ref-text "$REF_TEXT"

  for concurrency in $CONCURRENCIES; do
    output=$RUN_ROOT/${config}-c${concurrency}-seed-${SEED}
    if completed_result "$output"; then
      echo "[$config/C$concurrency] already complete; skipping"
      continue
    fi
    if [[ -e "$output" ]]; then
      archive=${output}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)
      mv "$output" "$archive"
      echo "[$config/C$concurrency] moved incomplete result to $archive"
    fi
    echo "[$config/C$concurrency] benchmark starting"
    TARGET=nari \
    BASE_URL=$BASE_URL \
    MODEL=yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF \
    VOICE=benchmark \
    CONCURRENCY=$concurrency \
    WARMUP=$WARMUP \
    DURATION=$DURATION \
    SEED=$SEED \
    OUTPUT=$output \
      bash scripts/run_fixed_concurrency_benchmark.sh
  done

  stop_server
  stop_mps
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader | sed -n '1p'
done

echo "Ablation complete: $RUN_ROOT"
