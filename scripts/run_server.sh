#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if [ ! -x "$FCV_VENV/bin/python" ]; then
  echo "Environment not found at $FCV_VENV; run bash scripts/setup_env.sh." >&2
  exit 1
fi
exec "$FCV_VENV/bin/python" -m faster_cosyvoice.server.app "$@"
