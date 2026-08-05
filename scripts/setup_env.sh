#!/usr/bin/env bash
# 分层复用 vllm025_venv（vllm 0.25.1 wheel，dspark 内置），不重装 vllm。
# 产出：本仓 venv/（py3.12 + datasets/soundfile/pytest + .pth 指向 vllm025 site-packages）
#       third_party/spec-vllm（yuekaizhang/vllm fork 源码，rep-penalty mirror，盖 PYTHONPATH）
set -euo pipefail
cd "$(dirname "$0")/.."

V025_SITE=${V025_SITE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/vllm025_venv/lib/python3.12/site-packages}
[ -d "$V025_SITE" ] || { echo "缺 $V025_SITE"; exit 1; }

# python3.12（uv 管理）
uv python install 3.12
if [ ! -d "$PWD/venv" ]; then
  uv venv --python 3.12 venv
fi
# 把 vllm025 的 site-packages 挂进我们的 venv（.pth，排在 PYTHONPATH 之后、可被 fork 覆盖）
echo "$V025_SITE" > venv/lib/python3.12/site-packages/_vllm025.pth
uv pip install --python venv/bin/python datasets soundfile pytest

# rep-penalty mirror（vllm PR #48932 合并前必需）：fork 源码盖在 wheel 之上
# github 不可达时可用本地镜像：
#   SPEC_VLLM_SRC=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/vllm bash scripts/setup_env.sh
SPEC_VLLM_DIR=$PWD/third_party/spec-vllm
SPEC_VLLM_SRC=${SPEC_VLLM_SRC:-https://github.com/yuekaizhang/vllm}
if [ ! -d "$SPEC_VLLM_DIR" ]; then
  git clone -b dspark-draft-sampling-mirrors "$SPEC_VLLM_SRC" "$SPEC_VLLM_DIR"
fi
find "$V025_SITE/vllm" -maxdepth 1 -name "*.so" -exec ln -sf {} "$SPEC_VLLM_DIR/vllm/" \;
# _version.py 由构建生成，fork 源码树里没有；同样从 wheel 软链（同用户本地镜像的做法）
ln -sf "$V025_SITE/vllm/_version.py" "$SPEC_VLLM_DIR/vllm/_version.py"

echo "环境就绪。使用前执行："
echo "  source venv/bin/activate"
echo "  export PYTHONPATH=$SPEC_VLLM_DIR:\$PYTHONPATH"
echo "  export HF_HOME=\${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface}"
