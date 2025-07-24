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

If ``--view_dependent`` is supplied, the script will compute
view‑dependent colours using the default camera position (see below).

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
    """Determine the number of frames in the motion sequence.

    The function uses the length of the first array in the motion
    dictionary to infer the number of frames.  If the motion dictionary
    is empty, ``0`` is returned.

    Args:
        motion: Dictionary of motion parameters.

    Returns:
        The inferred number of frames.
    """
    if not motion:
        return 0
    # Use the length of the first array to determine frame count
    first_key = next(iter(motion))
    return len(motion[first_key])


def export_sequence(model_dir: str, npz_path: str, output_dir: str,
                    view_dependent: bool = False) -> None:
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
    ckpt_path = _find_checkpoint(model_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    gaussians = GaussianModel()
    load_data = torch.load(ckpt_path, weights_only=False)
    # `restore` populates the model parameters; it returns self
    gaussians.restore(load_data)

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

    # Determine key names for pose, rotation and translation
    pose_key = None
    if 'pose' in motion:
        pose_key = 'pose'
    elif 'body_pose' in motion:
        pose_key = 'body_pose'
    # Keys for rotation and translation
    Rh_key = 'Rh' if 'Rh' in motion else ('global_orient' if 'global_orient' in motion else None)
    Th_key = 'Th' if 'Th' in motion else ('transl' if 'transl' in motion else None)
    # Optional facial parameters
    exp_key = 'expression' if 'expression' in motion else None
    jaw_key = 'jaw_pose' if 'jaw_pose' in motion else None

    for idx in range(n_frames):
        # Set pose parameters
        if pose_key is not None:
            pose_np = motion[pose_key][idx]
            # If using body_pose only (63 dims), prepend the global_orient to form 69 dims
            if pose_key == 'body_pose':
                if Rh_key is not None:
                    global_orient = motion[Rh_key][idx]
                else:
                    global_orient = np.zeros(3, dtype=np.float32)
                pose_np = np.concatenate([global_orient, pose_np], axis=0)
            # Convert to torch tensor
            pose_t = torch.from_numpy(pose_np).float()
            gaussians.smpl_poses = pose_t

        # Set global rotation (Rh)
        if Rh_key is not None:
            Rh_np = motion[Rh_key][idx]
            gaussians.Rh = torch.from_numpy(Rh_np).float()
        else:
            gaussians.Rh = torch.zeros(3, dtype=torch.float32)

        # Set global translation (Th)
        if Th_key is not None:
            Th_np = motion[Th_key][idx]
            gaussians.Th = torch.from_numpy(Th_np).float()
        else:
            gaussians.Th = torch.zeros(3, dtype=torch.float32)

        # Facial parameters
        if exp_key is not None:
            exp_np = motion[exp_key][idx]
            gaussians.expression = torch.from_numpy(exp_np).float()
        if jaw_key is not None:
            jaw_np = motion[jaw_key][idx]
            gaussians.jaw_pose = torch.from_numpy(jaw_np).float()

        # Compute output filename
        fname = f"frame_{idx:04d}.ply"
        fpath = os.path.join(output_dir, fname)
        # Compute camera position for view‑dependent colour if requested
        cam_pos = None
        if view_dependent:
            # Use origin as a default camera position; users may customise this
            # by editing this line or adding a command‑line argument.
            cam_pos = torch.zeros(3, dtype=torch.float32, device=device)
        # Export current frame
        gaussians.export_gaussians_to_ply(fpath, cam_pos=cam_pos)
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
                        help='Use view‑dependent colours (default: False)')
    args = parser.parse_args()
    export_sequence(args.model_dir, args.npz_path, args.output_dir,
                    view_dependent=args.view_dependent)


if __name__ == '__main__':
    main()