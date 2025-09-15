#!/usr/bin/env bash
set -euo pipefail

# Create a clean conda env for SMPL-X face optimization without touching system Python.
# Usage:
#   bash scripts/face_opt/env_conda.sh            # creates env 'mmlp_face'
#   bash scripts/face_opt/env_conda.sh myenvname  # custom name

ENV_NAME=${1:-mmlp_face}

echo "[conda] Creating env: ${ENV_NAME} (python=3.10, pip)"
conda create -n "${ENV_NAME}" -y python=3.10 pip

echo "[conda] Installing pinned NumPy/SciPy + essentials via pip"
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install \
  "numpy==1.26.4" \
  "scipy==1.11.4" \
  "typing_extensions>=4.8,<5"

echo "[conda] Installing CV/ML deps (headless OpenCV, MediaPipe, SMPL-X, Torch CPU)"
conda run -n "${ENV_NAME}" pip install \
  opencv-python-headless==4.10.0.84 \
  mediapipe==0.10.14 \
  smplx==0.1.28

# Torch CPU wheels from official index
conda run -n "${ENV_NAME}" pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

echo "[conda] Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "Run with (avoid user site-packages mixing):"
echo "  PYTHONNOUSERSITE=1 python -s -m scripts.face_opt.run_opt --subject /home/hello/data/SQ_02 --frame 932 --smpl_model_dir smpl_model/smplx --out_root output/face_opt --max_views 12"

