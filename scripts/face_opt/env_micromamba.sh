#!/usr/bin/env bash
set -euo pipefail

# Create an isolated micromamba env locally (no system pip/venv required).
# Usage:
#   bash scripts/face_opt/env_micromamba.sh
#   . .mamba/envs/smplx_face/bin/activate  # optional

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
MAMBA_ROOT="$ROOT_DIR/.mamba"
ENV_DIR="$MAMBA_ROOT/envs/smplx_face"

mkdir -p "$MAMBA_ROOT/bin"
if [[ ! -x "$MAMBA_ROOT/bin/micromamba" ]]; then
  echo "[env] Downloading micromamba..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xj -C "$MAMBA_ROOT/bin" --strip-components=1 bin/micromamba
fi

"$MAMBA_ROOT/bin/micromamba" --version

if [[ ! -d "$ENV_DIR" ]]; then
  echo "[env] Creating env at $ENV_DIR"
  "$MAMBA_ROOT/bin/micromamba" create -y -p "$ENV_DIR" -c conda-forge python=3.10 pip
fi

echo "[env] Installing Python packages via pip inside the isolated env"
"$ENV_DIR/bin/pip" install --upgrade pip
"$ENV_DIR/bin/pip" install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  opencv-python-headless==4.10.0.84 \
  mediapipe==0.10.14 \
  smplx==0.1.28 \
  torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu

echo "[env] Done. Activate with:"
echo "  source $ENV_DIR/bin/activate"
echo "Or run scripts with:"
echo "  $ENV_DIR/bin/python scripts/face_opt/run_opt.py ..."
