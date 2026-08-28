#!/usr/bin/env bash
# Backward-compatible wrapper. The canonical install command is now simply:
#   uv sync --frozen
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$FCV_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Remove only the two legacy overlay files created by the previous installer.
# A fresh uv environment never contains them.
LEGACY_SITE="$FCV_ROOT/.venv/lib/python3.12/site-packages"
if [ -f "$LEGACY_SITE/00_spec_vllm.pth" ]; then
  rm "$LEGACY_SITE/00_spec_vllm.pth"
fi
if [ -f "$LEGACY_SITE/01_vllm025.pth" ]; then
  rm "$LEGACY_SITE/01_vllm025.pth"
fi

uv sync --frozen

echo "Environment ready. Run:"
echo "  bash scripts/run_server.sh --help"
echo "  uv run python examples/offline_inference.py --help"
