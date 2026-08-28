#!/usr/bin/env bash
# Lint maintained serving/client surfaces, then run the CPU test suite.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$FCV_ROOT"
source scripts/env.sh

"$FCV_VENV/bin/ruff" check \
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
"$FCV_VENV/bin/python" -m pytest --ignore=tests/gpu
