#!/usr/bin/env bash
# Shared runtime environment. Source this file before running Python directly;
# scripts/run_server.sh sources it automatically.

FCV_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FCV_ROOT

if [ -z "${FCV_VENV:-}" ]; then
  if [ -x "$FCV_ROOT/.venv/bin/python" ]; then
    FCV_VENV="$FCV_ROOT/.venv"
  else
    # Compatibility with environments made before the uv project migration.
    FCV_VENV="$FCV_ROOT/venv"
  fi
fi
export FCV_VENV

FCV_CACHE_DIR=${FCV_CACHE_DIR:-$FCV_ROOT/.cache}
export FCV_CACHE_DIR
export HF_HOME=${HF_HOME:-$FCV_CACHE_DIR/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$FCV_CACHE_DIR}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$FCV_CACHE_DIR/flashinfer-workspace}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$FCV_CACHE_DIR/numba}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$FCV_CACHE_DIR/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$FCV_CACHE_DIR/triton}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# The layered base environment contains vLLM-Omni entry-point metadata but not
# its Python package. This service needs no external vLLM plugins; disabling
# auto-discovery avoids a noisy, harmless import error in every spawned worker.
export VLLM_PLUGINS=${VLLM_PLUGINS-}

# setup_env.sh also installs these paths through .pth files. PYTHONPATH keeps
# legacy venvs and direct source-tree invocations working.
SPEC_VLLM_DIR=${SPEC_VLLM_DIR:-$FCV_ROOT/third_party/spec-vllm}
if [ -d "$SPEC_VLLM_DIR/vllm" ]; then
  export PYTHONPATH="$SPEC_VLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p "$HF_HOME" "$NUMBA_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
