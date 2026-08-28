#!/usr/bin/env bash
# Run one independent Nari tts-bench rate point against faster-cosyvoice.
# Override settings with environment variables, for example:
#   RPS=6 SEED=1 DURATION=5m bash scripts/run_nari_benchmark.sh
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BENCH_VENV=${BENCH_VENV:-$FCV_ROOT/.benchmark-venv}
DATASET=${DATASET:-$FCV_ROOT/benchmarks/data/seed-tts-eval.jsonl}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
MODEL=${MODEL:-yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF}
VOICE=${VOICE:-benchmark}
LANGUAGE=${LANGUAGE:-English}
RPS=${RPS:-1}
SEED=${SEED:-0}
WARMUP=${WARMUP:-30s}
DURATION=${DURATION:-5m}
TIMEOUT=${TIMEOUT:-120s}
OUTPUT=${OUTPUT:-$FCV_ROOT/benchmarks/results/rps-${RPS}-seed-${SEED}}

if [ ! -x "$BENCH_VENV/bin/bench" ] || [ ! -f "$DATASET" ]; then
  echo "Run bash scripts/setup_nari_benchmark.sh first." >&2
  exit 1
fi
if [ -e "$OUTPUT" ]; then
  echo "Output already exists: $OUTPUT (tts-bench never reuses a result directory)." >&2
  exit 1
fi

exec "$BENCH_VENV/bin/bench" run \
  --target nari \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --voice "$VOICE" \
  --language "$LANGUAGE" \
  --dataset "$DATASET" \
  --rps "$RPS" \
  --warmup "$WARMUP" \
  --duration "$DURATION" \
  --arrival poisson \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --max-in-flight 4096 \
  --output "$OUTPUT" \
  "$@"
