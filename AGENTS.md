# Repository Guidelines

## Project Structure & Module Organization
- `scene/`: Core models and pipeline (e.g., `gaussian_model.py`, `scene.py`, `dataset.py`, MLPs, rendering utils).
- `utils/`: Math, I/O, config, and loss helpers.
- `config/`: YAML configs per dataset/subject (e.g., `actor01.yaml`).
- `viewer/`: Live viewer (`net_viewer.py`) and network glue.
- `assets/`, `docs/`, `script/`: Figures, notes, and helpers (e.g., `script/gen_weight_volume.py`).
- Top-level: `train.py`, `test.py`, `visualize.py`, export tools; `smpl_model/` for SMPL‑X files; `output/` for runs (git‑ignored).

## Build, Test, and Development Commands
- Create env (CUDA 12.1 example):
  - `conda create -n mmlphuman python=3.10 && conda activate mmlphuman`
  - Install deps per `README.md` (PyTorch 2.4.1, gsplat, PyTorch3D, etc.).
- Train: `python train.py --config config/{DATASET}.yaml --data_dir {DATASET_DIR} --out_dir output/{RUN}`
- Test/Render: `python test.py --config config/{DATASET}.yaml --model_dir output/{RUN} --out_dir output/{RUN}/images`
- Visualize (live): `cd viewer && python net_viewer.py` (connect via `visualize.py` or training)
- Weight volume: `cd script && python gen_weight_volume.py --data_dir {DATASET_DIR} --smpl_path ./smpl_model/smplx/SMPLX_NEUTRAL.npz`

## Coding Style & Naming Conventions
- Python 3.10, 4‑space indentation, PEP 8. Prefer explicit names (e.g., `gaussian_model`, `test_transform_consistency.py`).
- Use type hints where practical; keep functions under ~150 lines.
- Optional formatters: `black` and `ruff` if installed. Examples: `ruff .` and `black .`.

## Testing Guidelines
- No formal unit suite yet. Use scenario scripts:
  - Consistency: `python test_transform_consistency.py` (checks SMPL‑X and Gaussian transforms).
  - Rendering: `python test.py ...` on small subsets; adjust `config/*` `test` fields.
- When changing rendering/math, include small before/after images or PSNR in PR description (e.g., `output/{RUN}/images`).

## Commit & Pull Request Guidelines
- Commits: concise prefixes from history — `feat:`, `fix:`, `bugfix:`, `docs:`, `refactor:`. Example: `fix: handle expression and jaw_pose dims`.
- PRs must include: purpose, dataset/config used, exact commands run, and sample outputs (paths like `output/{RUN}/...`). Link related issues.

## Security & Configuration Tips
- Do not commit datasets or large artifacts; keep outputs under `output/` (git‑ignored).
- Place SMPL‑X model at `smpl_model/smplx/SMPLX_NEUTRAL.npz`.
- GPU required; training is long‑running. Set `seed` in configs for reproducibility.

