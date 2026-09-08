#!/usr/bin/env bash
# Run one fixed closed-loop concurrency point with the same Nari metrics/artifacts
# used by run_nari_benchmark.sh.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BENCH_VENV=${BENCH_VENV:-$FCV_ROOT/.benchmark-venv}
DATASET=${DATASET:-$FCV_ROOT/benchmarks/data/seed-tts-eval.jsonl}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
TARGET=${TARGET:-nari}
MODEL=${MODEL:-yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF}
VOICE=${VOICE:-benchmark}
LANGUAGE=${LANGUAGE:-English}
CONCURRENCY=${CONCURRENCY:-1}
SEED=${SEED:-0}
WARMUP=${WARMUP:-15s}
DURATION=${DURATION:-60s}
TIMEOUT=${TIMEOUT:-120s}
OUTPUT=${OUTPUT:-$FCV_ROOT/benchmarks/results/fixed-concurrency-${CONCURRENCY}-seed-${SEED}}

if [ ! -x "$BENCH_VENV/bin/python" ] || [ ! -f "$DATASET" ]; then
  echo "Run bash scripts/setup_nari_benchmark.sh first." >&2
  exit 1
fi
if [ -e "$OUTPUT" ]; then
  echo "Output already exists: $OUTPUT (benchmark evidence is never reused)." >&2
  exit 1
fi

exec "$BENCH_VENV/bin/python" "$FCV_ROOT/scripts/fixed_concurrency_client.py" \
  --target "$TARGET" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --voice "$VOICE" \
  --language "$LANGUAGE" \
  --dataset "$DATASET" \
  --concurrency "$CONCURRENCY" \
  --warmup "$WARMUP" \
  --duration "$DURATION" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --output "$OUTPUT" \
  "$@"
