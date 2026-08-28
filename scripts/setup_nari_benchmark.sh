#!/usr/bin/env bash
# Install Nari Labs' exact tts-bench lockfile into an isolated local venv and
# prepare its pinned 1,088-row English Seed-TTS prompt projection.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
NARI_BENCH_DIR=${NARI_BENCH_DIR:-$FCV_ROOT/../nari/benchmarks/tts_bench}
BENCH_VENV=${BENCH_VENV:-$FCV_ROOT/.benchmark-venv}
DATASET=${DATASET:-$FCV_ROOT/benchmarks/data/seed-tts-eval.jsonl}

if [ ! -f "$NARI_BENCH_DIR/uv.lock" ]; then
  echo "Nari tts-bench not found at $NARI_BENCH_DIR." >&2
  echo "Set NARI_BENCH_DIR=/path/to/benchmarks/tts_bench." >&2
  exit 1
fi

mkdir -p "$FCV_ROOT/.cache/uv" "$(dirname "$DATASET")"
UV_CACHE_DIR="$FCV_ROOT/.cache/uv" \
UV_PROJECT_ENVIRONMENT="$BENCH_VENV" \
  uv sync --project "$NARI_BENCH_DIR" --frozen --no-dev

if [ ! -f "$DATASET" ]; then
  "$BENCH_VENV/bin/bench" dataset prepare --output "$DATASET"
fi

echo "Nari benchmark ready:"
echo "  executable: $BENCH_VENV/bin/bench"
echo "  dataset:    $DATASET"
