#!/usr/bin/env bash
# Shared runtime environment. Source this file before running Python directly;
# scripts/run_server.sh sources it automatically.

FCV_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FCV_ROOT

# Some base images export UV_PROJECT_ENVIRONMENT for their own /opt venv.
# Never let that generic setting redirect this repository's `uv sync` away
# from the explicit faster-cosyvoice environment.
FCV_VENV=${FCV_VENV:-$FCV_ROOT/.venv}
export FCV_VENV
export UV_PROJECT_ENVIRONMENT=$FCV_VENV

FCV_CACHE_DIR=${FCV_CACHE_DIR:-$FCV_ROOT/.cache}
export FCV_CACHE_DIR
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FCV_CACHE_DIR/uv}
export HF_HOME=${HF_HOME:-$FCV_CACHE_DIR/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$FCV_CACHE_DIR}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$FCV_CACHE_DIR/flashinfer-workspace}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$FCV_CACHE_DIR/numba}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$FCV_CACHE_DIR/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$FCV_CACHE_DIR/triton}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# This service needs no external vLLM plugins. Disable auto-discovery so an
# unrelated globally registered plugin cannot affect a reproducible uv env.
export VLLM_PLUGINS=${VLLM_PLUGINS-}

mkdir -p "$HF_HOME" "$NUMBA_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$UV_CACHE_DIR"
