# Repository Guidelines

## Project Structure & Module Organization
- `scene/`: Core models and pipeline (`gaussian_model.py`, `scene.py`, `dataset.py`, MLPs, rendering helpers).
- `utils/`: Math, I/O, config, and loss utilities.
- `config/`: YAML configs per dataset/subject (e.g., `actor01.yaml`, `subject02.yaml`).
- `viewer/`: Live viewer (`net_viewer.py`) and network glue.
- `assets/`, `docs/`, `script/`: Figures, notes, and helper scripts (e.g., `script/gen_weight_volume.py`).
- Top-level entry points: `train.py`, `test.py`, `visualize.py`, export tools, plus `smpl_model/` for SMPL-X files and `output/` for runs.

## Build, Test, and Development Commands
- Create env (CUDA 12.1 example):
  - `conda create -n mmlphuman python=3.10 && conda activate mmlphuman`
  - Install deps as in `README.md` (torch 2.4.1, gsplat, pytorch3d, etc.).
- Train: `python train.py --config config/{DATASET}.yaml --data_dir {DATASET_DIR} --out_dir output/{RUN}`
- Test/render: `python test.py --config config/{DATASET}.yaml --model_dir output/{RUN} --out_dir output/{RUN}/images`
- Visualize (live): `cd viewer && python net_viewer.py`; connect via `visualize.py` or training.
- Weight volume: `cd script && python gen_weight_volume.py --data_dir {DATASET_DIR} --smpl_path ./smpl_model/smplx/SMPLX_NEUTRAL.npz`

## Coding Style & Naming Conventions
- Python 3.10, 4‑space indentation, PEP 8. Prefer explicit names (`gaussian_model`, `test_transform_consistency.py`).
- Type hints where practical; keep functions <150 lines.
- Optional formatters: `black` and `ruff` (if available). Avoid unrelated refactors in single PRs.

## Testing Guidelines
- No formal unit test suite yet. Use scenario scripts:
  - Consistency check: `python test_transform_consistency.py` (verifies SMPL-X and Gaussian transforms).
  - Rendering tests via `test.py` with small subsets (adjust `config/*` `test` fields).
- Include small before/after images or PSNR in PR description when changing rendering/math.

## Commit & Pull Request Guidelines
- Commit style: use concise prefixes seen in history: `feat:`, `fix:`, `bugfix:`, `docs:`, `refactor:`. Example: `fix: handle expression and jaw_pose dims`.
- PRs must include: purpose, dataset/config used, commands run, and sample outputs (paths like `output/{RUN}/...`). Link issues when applicable.

## Security & Configuration Tips
- Do not commit datasets or large artifacts; outputs go under `output/` and are git‑ignored.
- Place SMPL‑X model at `smpl_model/smplx/SMPLX_NEUTRAL.npz`.
- GPU required; training is long‑running. Set `seed` in configs for reproducibility.

