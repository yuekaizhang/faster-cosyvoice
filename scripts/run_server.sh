#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=$PWD/third_party/spec-vllm:$PYTHONPATH
export HF_HOME=${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface}
exec python -m faster_cosyvoice.server.app "$@"
