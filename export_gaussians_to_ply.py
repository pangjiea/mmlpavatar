"""
Export Gaussians to PLY sequence.

This script provides a simple command‑line interface for exporting the
Gaussian points contained in a trained model to a sequence of PLY files.
The model is loaded from a checkpoint directory and posed according to a
motion sequence stored in a ``.npz`` file.  The resulting PLY files are
written to an output directory with one file per frame.

Example usage::

    python export_gaussians_to_ply.py \
        --model_dir path/to/checkpoint_dir \
        --npz_path path/to/motion.npz \
        --output_dir path/to/output/plys

Options:

* ``--format standard|simple``: output PLY format (default: ``standard``).
* ``--color_mode sh|position|opacity|uniform``: for ``simple`` format.
* ``--view_dependent``: compute view‑dependent colours (forces simple PLY).

The ``.npz`` file is expected to contain at least the following keys:

* ``pose`` or ``body_pose``: body pose parameters (63 values).  If
  ``body_pose`` is provided, a matching ``global_orient`` (3 values)
  should also be present.  These two arrays will be concatenated to
  form a 69‑dimensional pose vector for SMPL/SMPLX.
* ``Rh`` or ``global_orient``: axis‑angle rotation vectors (3 values).
* ``Th`` or ``transl``: translations (3 values).
* Optionally ``expression`` (10 values) and ``jaw_pose`` (3 values).

If some of these keys are missing they will default to zero.
"""

import argparse
import os
import sys
import numpy as np
import torch

# Ensure the package root is on the path when running from a cloned repo
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scene.gaussian_model import GaussianModel  # type: ignore


def _write_ply_points(path: str, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """Write a simple ASCII PLY file with XYZ and optional RGB.

    Args:
        path: Output file path.
        xyz: (N,3) float32 array.
        rgb: Optional (N,3) uint8 array.
    """
    n = int(xyz.shape[0])
    has_color = rgb is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header += ["end_header"]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if has_color:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z) in xyz:
                f.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")


def _fallback_export_simple(gaussians: GaussianModel, out_path: str, color_mode: str = "sh",
                            cam_pos: torch.Tensor | None = None) -> None:
    """Fallback exporter that writes a simple point cloud PLY (XYZ [+ RGB]).

    Color modes:
      - sh: use SH via gaussians.get_color(cam_pos) with cam_pos default 0.
      - opacity: gray from get_opacity.
      - position: min-max normalised XYZ to RGB.
      - uniform: constant grey.
    """
    # Get world positions
    xyz_t = gaussians.get_xyz
    if xyz_t.is_cuda:
        xyz = xyz_t.detach().cpu().numpy().astype(np.float32)
    else:
        xyz = xyz_t.detach().numpy().astype(np.float32)

    rgb = None
    try:
        if color_mode == "sh":
            if cam_pos is None:
                device = xyz_t.device if isinstance(xyz_t, torch.Tensor) else torch.device("cpu")
                cam_pos = torch.zeros(3, dtype=torch.float32, device=device)
            col = gaussians.get_color(cam_pos)
            col = torch.clamp(col, 0.0, 1.0)
            rgb = (col.detach().cpu().numpy() * 255.0).astype(np.uint8)
        elif color_mode == "opacity":
            op = gaussians.get_opacity
            g = torch.clamp(op, 0.0, 1.0).detach().cpu().numpy()
            g = (g * 255.0).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=1)
        elif color_mode == "position":
            mn = xyz.min(axis=0)
            mx = xyz.max(axis=0)
            span = np.maximum(mx - mn, 1e-6)
            col = (xyz - mn) / span
            rgb = (np.clip(col, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif color_mode == "uniform":
            rgb = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
    except Exception:
        # If anything fails, fall back to no color
        rgb = None

    _write_ply_points(out_path, xyz, rgb)
from utils.smpl_utils import init_smpl_pose  # ensure smpl.smpl_bigpose is available for restore()


def _find_checkpoint(model_dir: str) -> str:
    """Locate the most recent checkpoint (.pth) file in ``model_dir``.

    The function first looks for files containing both ``'chkpnt'`` and
    ``'.pth'`` (following the convention used in the original code).  If
    none are found, it falls back to any ``.pth`` file in the directory.

    Args:
        model_dir: Path to the directory containing checkpoints.

    Returns:
        The absolute path to the selected checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint file could be found.
    """
    candidates = [f for f in os.listdir(model_dir)
                  if 'chkpnt' in f and f.endswith('.pth')]
    if not candidates:
        # Fallback to any .pth files
        candidates = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint found in {model_dir}")
    candidates.sort(key=lambda s: (len(s), s))
    ckpt_path = os.path.join(model_dir, candidates[-1])
    return ckpt_path


def _load_motion(npz_path: str) -> dict:
    """Load a motion sequence from an NPZ file.

    The returned dictionary contains numpy arrays keyed by the expected
    parameter names.  Missing keys are not added; the caller should
    handle defaults.

    Args:
        npz_path: Path to the .npz file.

    Returns:
        A dictionary mapping string keys to numpy arrays.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Motion npz file not found: {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def _get_frame_count(motion: dict) -> int:
    """Determine number of frames from all arrays, robustly.

    Prefer arrays whose first dimension > 1; otherwise fall back to 1.
    """
    if not motion:
        return 0
    counts = []
    for k, v in motion.items():
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.ndim >= 1:
            counts.append(int(a.shape[0]))
    if not counts:
        return 0
    # Prefer any count > 1, else max (which may be 1)
    mx = max(counts)
    return int(mx)


def _select_frame(v: np.ndarray, idx: int) -> np.ndarray:
    """Select frame idx from array v safely.

    - If v is a vector with shape in {3,10,45,63,165}, treat as constant across frames.
    - If idx >= len(v), clamp to last frame.
    """
    a = np.asarray(v)
    if a.ndim == 0:
        return a
    if a.ndim == 1 and a.shape[0] in (3, 10, 45, 63, 165):
        return a
    # sequence-like
    i = min(idx, a.shape[0] - 1)
    return a[i]


def _compose_full_pose(motion: dict, idx: int) -> np.ndarray:
    """Compose a 165-dim SMPL-X pose vector for frame ``idx``.

    Order: [global_orient(3), body_pose(63), jaw(3), leye(3), reye(3),
            left_hand(45), right_hand(45)] = 165.
    Missing parts are padded with zeros.
    If a 'pose' array of length 165 already exists, it is returned as-is.
    """
    # Direct 'pose' path
    if 'pose' in motion:
        pose = motion['pose'][idx]
        if pose.shape[-1] == 165:
            return pose.astype(np.float32)
    # Collect components
    def get(name, d):
        return motion[name][idx] if name in motion else np.zeros(d, dtype=np.float32)
    global_orient = get('global_orient', 3)
    body_pose = motion['body_pose'][idx] if 'body_pose' in motion else (
        motion['pose'][idx][3:3+63] if 'pose' in motion and motion['pose'][idx].shape[-1] >= 66 else np.zeros(63, dtype=np.float32)
    )
    jaw_pose = get('jaw_pose', 3)
    leye_pose = get('leye_pose', 3)
    reye_pose = get('reye_pose', 3)
    left_hand = get('left_hand_pose', 45)
    right_hand = get('right_hand_pose', 45)
    parts = [global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand, right_hand]
    pose = np.concatenate(parts, axis=0).astype(np.float32)
    return pose


def export_sequence(model_dir: str, npz_path: str, output_dir: str,
                    view_dependent: bool = False,
                    ply_format: str = 'standard',
                    color_mode: str = 'sh',
                    scale_mode: str = 'auto',
                    scale_factor: float = 1.0,
                    scale_min: float | None = None,
                    scale_max: float | None = None,
                    opacity_mode: str = 'auto',
                    only_frame: int | None = None) -> None:
    """Export a sequence of PLY files from a Gaussian model.

    Args:
        model_dir: Directory containing the trained model checkpoint(s).
        npz_path: Path to the motion parameters stored as an ``.npz`` file.
        output_dir: Directory where the PLY files should be written.  This
            directory will be created if it does not already exist.
        view_dependent: If ``True``, colours will be computed
            view‑dependently using a default camera position (the origin).

    The function iterates over all frames in the motion sequence, sets the
    corresponding pose parameters on the Gaussian model, and calls
    ``export_gaussians_to_ply`` for each frame.  Filenames are zero‑padded
    sequential numbers starting at 0 (e.g., ``frame_0000.ply``).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load Gaussian model from checkpoint
    # Important: GaussianModel.restore() expects smpl.smpl_bigpose to be initialized.
    # We don't need the full SMPL-X model here; init_smpl_pose() is sufficient.
    init_smpl_pose()
    ckpt_path = _find_checkpoint(model_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    gaussians = GaussianModel()
    load_data = torch.load(ckpt_path, weights_only=False)
    # `restore` populates the model parameters; it returns self
    gaussians.restore(load_data)
    # Ensure encoder params are float32 to avoid dtype mismatch
    if hasattr(gaussians, 'encoder_feat_params') and isinstance(gaussians.encoder_feat_params, dict):
        for k in list(gaussians.encoder_feat_params.keys()):
            gaussians.encoder_feat_params[k] = gaussians.encoder_feat_params[k].float()

    # The Gaussian model stores parameters on CPU by default; if CUDA is
    # available we move the model to GPU for faster computation.  This
    # implicitly moves all learnable parameters; non‑learnable buffers
    # accessed during export (e.g., via `.cuda(non_blocking=True)`) will
    # allocate GPU memory as needed.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    motion = _load_motion(npz_path)
    n_frames = _get_frame_count(motion)
    if n_frames == 0:
        print("No frames found in motion file.")
        return

    # Keys for rotation and translation
    Rh_key = 'Rh' if 'Rh' in motion else ('global_orient' if 'global_orient' in motion else None)
    Th_key = 'Th' if 'Th' in motion else ('transl' if 'transl' in motion else None)
    # Optional facial parameters
    exp_key = 'expression' if 'expression' in motion else None
    jaw_key = 'jaw_pose' if 'jaw_pose' in motion else None
    leye_key = 'leye_pose' if 'leye_pose' in motion else None
    reye_key = 'reye_pose' if 'reye_pose' in motion else None

    if only_frame is not None:
        if only_frame < 0 or only_frame >= n_frames:
            print(f"Requested frame {only_frame} is out of range [0, {n_frames-1}] — skipping.")
            return
        frame_indices = [only_frame]
    else:
        frame_indices = list(range(n_frames))

    for idx in frame_indices:
        # Set full pose (165) to satisfy joint assertions in GaussianModel
        pose_np = _compose_full_pose(motion, idx)
        gaussians.smpl_poses = torch.from_numpy(pose_np).float()

        # Set global rotation (Rh)
        if Rh_key is not None:
            Rh_np = _select_frame(motion[Rh_key], idx)
            # Accept either axis-angle (3,) or rotation matrix (3x3)
            if Rh_np.shape == (3,):
                try:
                    from scipy.spatial.transform import Rotation as R
                    Rh_mat = R.from_rotvec(Rh_np.astype(np.float32)).as_matrix().astype(np.float32)
                except Exception:
                    # Minimal Rodrigues fallback
                    theta = np.linalg.norm(Rh_np.astype(np.float32) + 1e-8)
                    if theta < 1e-8:
                        Rh_mat = np.eye(3, dtype=np.float32)
                    else:
                        k = Rh_np.astype(np.float32) / theta
                        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
                        I = np.eye(3, dtype=np.float32)
                        Rh_mat = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                gaussians.Rh = torch.from_numpy(Rh_mat).float()
            elif Rh_np.shape == (3, 3):
                gaussians.Rh = torch.from_numpy(Rh_np.astype(np.float32)).float()
            else:
                # Unknown format; default to identity
                gaussians.Rh = torch.eye(3, dtype=torch.float32)
        else:
            # Identity rotation (setter will treat identity as None)
            gaussians.Rh = torch.eye(3, dtype=torch.float32)

        # Set global translation (Th)
        if Th_key is not None:
            Th_np = _select_frame(motion[Th_key], idx)
            gaussians.Th = torch.from_numpy(Th_np).float()
        else:
            gaussians.Th = torch.zeros(3, dtype=torch.float32)

        # Facial parameters
        if exp_key is not None:
            exp_np = np.asarray(_select_frame(motion[exp_key], idx), dtype=np.float32).reshape(-1)
            # Normalize to 10 dims to match training/config
            if exp_np.shape[0] > 10:
                exp_np = exp_np[:10]
            elif exp_np.shape[0] < 10:
                exp_np = np.pad(exp_np, (0, 10 - exp_np.shape[0]))
            gaussians.expression = torch.from_numpy(exp_np).float()
        if jaw_key is not None:
            jaw_np = np.asarray(_select_frame(motion[jaw_key], idx), dtype=np.float32).reshape(-1)
            if jaw_np.shape[0] > 3:
                jaw_np = jaw_np[:3]
            elif jaw_np.shape[0] < 3:
                jaw_np = np.pad(jaw_np, (0, 3 - jaw_np.shape[0]))
            gaussians.jaw_pose = torch.from_numpy(jaw_np).float()
        if leye_key is not None:
            leye_np = np.asarray(_select_frame(motion[leye_key], idx), dtype=np.float32).reshape(-1)
            if leye_np.shape[0] > 3:
                leye_np = leye_np[:3]
            elif leye_np.shape[0] < 3:
                leye_np = np.pad(leye_np, (0, 3 - leye_np.shape[0]))
            gaussians.leye_pose = torch.from_numpy(leye_np).float()
        if reye_key is not None:
            reye_np = np.asarray(_select_frame(motion[reye_key], idx), dtype=np.float32).reshape(-1)
            if reye_np.shape[0] > 3:
                reye_np = reye_np[:3]
            elif reye_np.shape[0] < 3:
                reye_np = np.pad(reye_np, (0, 3 - reye_np.shape[0]))
            gaussians.reye_pose = torch.from_numpy(reye_np).float()

        # Compute output filename
        fname = f"frame_{idx:04d}.ply"
        fpath = os.path.join(output_dir, fname)
        # Compute camera position for view‑dependent colour if requested
        cam_pos = None
        if view_dependent:
            # Use origin as a default camera position; users may customise this
            # by editing this line or adding a command‑line argument.
            cam_pos = torch.zeros(3, dtype=torch.float32, device=device)
        # Export current frame (use model's method when available; else fallback to simple PLY)
        if hasattr(gaussians, 'export_gaussians_to_ply') and callable(getattr(gaussians, 'export_gaussians_to_ply')):
            if cam_pos is not None:
                gaussians.export_gaussians_to_ply(fpath, cam_pos=cam_pos)
            else:
                # Determine modes based on format and user override
                std = (ply_format == 'standard')
                eff_scale_mode = ('log' if std else 'linear') if scale_mode == 'auto' else scale_mode
                eff_opacity_mode = ('logit' if std else 'alpha') if opacity_mode == 'auto' else opacity_mode
                kwargs = dict()
                if std:
                    kwargs.update(dict(scale_mode=eff_scale_mode,
                                       scale_factor=scale_factor,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       opacity_mode=eff_opacity_mode))
                else:
                    # compat/simple use linear scale controls
                    kwargs.update(dict(scale_factor=scale_factor,
                                       scale_min=scale_min if scale_min is not None else (1e-6 if ply_format=='compat' else None),
                                       scale_max=scale_max if scale_max is not None else (5e-3 if ply_format=='compat' else None)))
                gaussians.export_gaussians_to_ply(fpath, cam_pos=None, format_type=ply_format, color_mode=color_mode, **kwargs)
        else:
            if cam_pos is None and view_dependent:
                # ensure cam_pos for SH coloring if requested
                cam_pos = torch.zeros(3, dtype=torch.float32, device=device)
            _fallback_export_simple(gaussians, fpath, color_mode=('sh' if color_mode else 'uniform'), cam_pos=cam_pos)
        print(f"Saved PLY for frame {idx} to {fpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gaussian points to PLY sequence")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to motion parameters (.npz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for PLY files')
    parser.add_argument('--view_dependent', action='store_true',
                        help='Use view‑dependent colours (forces simple PLY)')
    parser.add_argument('--format', dest='ply_format', choices=['standard','simple','compat'], default='standard',
                        help='PLY format: standard GS or simple point cloud')
    parser.add_argument('--color_mode', choices=['sh','position','opacity','uniform'], default='sh',
                        help='Color mode for simple PLY')
    parser.add_argument('--scale_mode', choices=['auto','log','linear'], default='auto',
                        help='Scale encoding for PLY scale_* fields')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Multiply linear scales by this factor')
    parser.add_argument('--scale_min', type=float, default=None, help='Clamp linear scale min (None to disable)')
    parser.add_argument('--scale_max', type=float, default=None, help='Clamp linear scale max (None to disable)')
    parser.add_argument('--opacity_mode', choices=['auto','logit','alpha'], default='auto',
                        help='Opacity encoding for PLY opacity field')
    parser.add_argument('--frame', type=int, default=None, help='Export only this frame index (0-based)')
    args = parser.parse_args()
    export_sequence(args.model_dir, args.npz_path, args.output_dir,
                    view_dependent=args.view_dependent,
                    ply_format=args.ply_format,
                    color_mode=args.color_mode,
                    scale_mode=args.scale_mode,
                    scale_factor=args.scale_factor,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                    opacity_mode=args.opacity_mode,
                    only_frame=args.frame)


if __name__ == '__main__':
    main()
