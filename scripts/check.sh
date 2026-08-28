#!/usr/bin/env bash
# Lint maintained serving/client surfaces, then run the CPU test suite.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$FCV_ROOT"
source scripts/env.sh

uv run --frozen ruff check \
  faster_cosyvoice/config.py \
  faster_cosyvoice/server \
  faster_cosyvoice/streaming \
  examples/register_voice.py \
  examples/stream_client.py \
  tests/test_batcher.py \
  tests/test_config.py \
  tests/test_leading_silence.py \
  tests/test_protocol.py \
  tests/test_session.py \
  tests/test_stream_client.py
uv run --frozen python -m pytest --ignore=tests/gpu
