#!/usr/bin/env bash
# Reproducible thin project environment layered over the prebuilt vLLM 0.25.1
# environment. The expensive CUDA stack remains in VLLM_BASE_VENV; uv owns the
# Python version, project package, and lightweight dependencies in .venv.
set -euo pipefail

FCV_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$FCV_ROOT"

PYTHON_VERSION=${PYTHON_VERSION:-3.12.13}
FCV_VENV=${FCV_VENV:-$FCV_ROOT/.venv}
VLLM_BASE_VENV=${VLLM_BASE_VENV:-$FCV_ROOT/../vllm025_venv}

if [ -z "${V025_SITE:-}" ]; then
  V025_SITE=$(find "$VLLM_BASE_VENV/lib" -maxdepth 2 -type d \
    -path '*/python3.12/site-packages' -print -quit 2>/dev/null || true)
fi
if [ -z "$V025_SITE" ] || [ ! -f "$V025_SITE/vllm/__init__.py" ]; then
  echo "Missing the vLLM 0.25.1 base environment." >&2
  echo "Set VLLM_BASE_VENV or V025_SITE, for example:" >&2
  echo "  VLLM_BASE_VENV=/path/to/vllm025_venv bash scripts/setup_env.sh" >&2
  exit 1
fi

mkdir -p "$FCV_ROOT/.cache/uv"
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FCV_ROOT/.cache/uv}
if ! uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
  uv python install "$PYTHON_VERSION"
fi
UV_PROJECT_ENVIRONMENT="$FCV_VENV" \
  uv sync --frozen --extra quality --extra test

# Repetition-penalty mirror support lives in this fork until the corresponding
# vLLM change is available in the base wheel. Prefer a nearby local checkout
# when present; otherwise clone the public branch.
SPEC_VLLM_DIR=${SPEC_VLLM_DIR:-$FCV_ROOT/third_party/spec-vllm}
if [ -z "${SPEC_VLLM_SRC:-}" ]; then
  LOCAL_SPEC_VLLM="$FCV_ROOT/../../speculative/vllm"
  if [ -f "$LOCAL_SPEC_VLLM/vllm/__init__.py" ]; then
    SPEC_VLLM_SRC=$LOCAL_SPEC_VLLM
  else
    SPEC_VLLM_SRC=https://github.com/yuekaizhang/vllm
  fi
fi
if [ -e "$SPEC_VLLM_DIR" ] && [ ! -f "$SPEC_VLLM_DIR/vllm/__init__.py" ]; then
  echo "$SPEC_VLLM_DIR exists but is not a valid vLLM checkout; move it aside first." >&2
  exit 1
fi
if [ ! -f "$SPEC_VLLM_DIR/vllm/__init__.py" ]; then
  mkdir -p "$FCV_ROOT/third_party"
  CLONE_PARENT=$(mktemp -d "$FCV_ROOT/third_party/spec-vllm-clone.XXXXXX")
  trap 'rm -rf "$CLONE_PARENT"' EXIT
  git clone --branch dspark-draft-sampling-mirrors --single-branch \
    "$SPEC_VLLM_SRC" "$CLONE_PARENT/repo"
  mv "$CLONE_PARENT/repo" "$SPEC_VLLM_DIR"
fi

PROJECT_SITE=$(
  "$FCV_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
printf '%s\n' "$SPEC_VLLM_DIR" > "$PROJECT_SITE/00_spec_vllm.pth"
printf '%s\n' "$V025_SITE" > "$PROJECT_SITE/01_vllm025.pth"

# The fork is Python source over the prebuilt wheel. Link wheel-only extensions
# and generated metadata into the fork so it remains first on sys.path.
find "$V025_SITE/vllm" -maxdepth 1 -name '*.so' \
  -exec ln -sf {} "$SPEC_VLLM_DIR/vllm/" \;
FA_DIR="$SPEC_VLLM_DIR/vllm/vllm_flash_attn"
for item in _vllm_fa2_C.abi3.so _vllm_fa3_C.abi3.so cute layers ops; do
  if [ -e "$V025_SITE/vllm/vllm_flash_attn/$item" ]; then
    ln -sfn "$V025_SITE/vllm/vllm_flash_attn/$item" "$FA_DIR/$item"
  fi
done
ln -sf "$V025_SITE/vllm/_version.py" "$SPEC_VLLM_DIR/vllm/_version.py"

echo "Environment ready. Run:"
echo "  source scripts/env.sh"
echo "  source \"\$FCV_VENV/bin/activate\""
echo "  faster-cosyvoice-server --help"
