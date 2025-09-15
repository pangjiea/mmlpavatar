#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Chamfer distance between predicted PLY frames and SMPL-X meshes in world coordinates.

One-click usage (defaults try to work with this repo layout):

  python scripts/eval_ply_vs_smplx.py \
      --ply_dir /home/hello/code/mmlphuman/render/zzr_test1700/plys \
      --smpl_npz viewer/output/snapshots/ori_smpl_params.npz \
      --out ROOT/metrics_eval_smplx.json

Notes
  - SMPL-X model is loaded from './smpl_model/smplx' by default.
  - The npz is expected to contain per-frame parameters, e.g. either a combined
    'pose' (165) or separate keys like 'global_orient', 'body_pose',
    'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose';
    plus 'Th' or 'transl' for translation, 'Rh' (optional world rotation), and 'betas'.
  - Frame index is parsed from PLY filenames like 'frame_1701.ply'. The script uses
    that index to select the corresponding SMPL-X parameters.
  - Chamfer uses SciPy cKDTree; please install scipy.
  - You can uniformly sample points on the SMPL-X surface via --smpl_sample_points.
"""

import argparse
import os
from pathlib import Path
import json
import re
from typing import Dict, Tuple, Optional, List

import numpy as np


def try_import_smplx():
    try:
        import smplx  # type: ignore
        return smplx
    except Exception as e:
        raise RuntimeError(f"smplx not available: {e}")


def try_import_scipy():
    try:
        from scipy.spatial import cKDTree  # type: ignore
        return cKDTree
    except Exception as e:
        raise RuntimeError(f"scipy not available: {e}")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"npz not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def parse_frame_id(ply_name: str) -> Optional[int]:
    # Accept frame_XXXX.ply or just digits
    m = re.search(r'(?:frame_)?(\d+)\.ply$', ply_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_frames_from_dir(ply_dir: Path) -> List[Tuple[int, Path]]:
    frames = []
    for p in sorted(ply_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == '.ply':
            idx = parse_frame_id(p.name)
            if idx is not None:
                frames.append((idx, p))
    return frames


def load_ply_points_ascii(ply_path: Path) -> np.ndarray:
    """Minimal ASCII PLY reader for vertex-only files with x,y,z[, ...]."""
    # Confirm ASCII quickly
    with open(ply_path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line == b'':
                break
            if line.strip() == b'end_header':
                break
        header_txt = b''.join(header).decode('utf-8', errors='ignore')
        if 'format ascii' not in header_txt:
            raise RuntimeError('Binary PLY; please install open3d or trimesh to read it')
    # Load ASCII data
    data = np.loadtxt(ply_path, dtype=np.float32, comments=['ply', 'format', 'element', 'property', 'comment', 'obj_info', 'end_header'])
    if data.ndim == 1:
        data = data[None, :]
    pts = data[:, :3].astype(np.float32)
    return pts


def chamfer(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    cKDTree = try_import_scipy()
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {"chamfer_L2_cm2": float('nan'), "chamfer_L2_cm": float('nan')}
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1, workers=-1)
    ta = cKDTree(a)
    db, _ = ta.query(b, k=1 ,workers=-1)
    da_cm = da * 100.0
    db_cm = db * 100.0
    c2 = 0.5 * (np.mean(da_cm**2) + np.mean(db_cm**2))
    c1 = 0.5 * (np.mean(da_cm) + np.mean(db_cm))
    return {"chamfer_L2_cm2": float(c2), "chamfer_L2_cm": float(c1)}


def compose_pose165_from_npz(motion: Dict[str, np.ndarray], idx: int) -> np.ndarray:
    # Direct
    if 'pose' in motion:
        pose = np.asarray(motion['pose'])
        if pose.ndim == 1:
            return pose.astype(np.float32)
        if pose.shape[-1] == 165:
            i = min(idx, pose.shape[0]-1)
            return pose[i].astype(np.float32)
    def sel(name: str, d: int) -> np.ndarray:
        if name not in motion:
            return np.zeros(d, dtype=np.float32)
        arr = np.asarray(motion[name])
        if arr.ndim == 1 and arr.shape[0] == d:
            return arr.astype(np.float32)
        i = min(idx, arr.shape[0]-1)
        return arr[i].astype(np.float32)
    global_orient = sel('global_orient', 3)
    body_pose = sel('body_pose', 63)
    jaw_pose = sel('jaw_pose', 3)
    leye_pose = sel('leye_pose', 3)
    reye_pose = sel('reye_pose', 3)
    left_hand = sel('left_hand_pose', 45)
    right_hand = sel('right_hand_pose', 45)
    return np.concatenate([global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand, right_hand], axis=0).astype(np.float32)


def select_frame(arr: Optional[np.ndarray], idx: int, d: Optional[int] = None) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 1:
        if d is not None and a.shape[0] != d:
            # pad or trim
            if a.shape[0] > d:
                a = a[:d]
            else:
                a = np.pad(a, (0, d - a.shape[0]))
        return a.astype(np.float32)
    i = min(idx, a.shape[0]-1)
    out = a[i]
    if d is not None and out.shape[0] != d:
        if out.shape[0] > d:
            out = out[:d]
        else:
            out = np.pad(out, (0, d - out.shape[0])) 
    return out.astype(np.float32)


def _sample_points_on_mesh(vertices: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample n points on a triangle mesh surface using area weights."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # Triangle areas
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate; fallback to vertices
        idx = np.random.randint(0, vertices.shape[0], size=int(n))
        return vertices[idx]
    prob = areas / total
    tri_idx = np.random.choice(len(faces), size=int(n), p=prob)
    v0 = v0[tri_idx]
    v1 = v1[tri_idx]
    v2 = v2[tri_idx]
    r1 = np.random.rand(int(n), 1).astype(np.float32)
    r2 = np.random.rand(int(n), 1).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    pts = w0 * v0 + w1 * v1 + w2 * v2
    return pts.astype(np.float32)


def smplx_points_world(smplx_model, pose165: np.ndarray, beta: Optional[np.ndarray], Th: Optional[np.ndarray], Rh: Optional[np.ndarray], sample_points: int = 0):
    """Build torch tensors for SMPL-X call following dataset convention.

    - pose165 stores global_orient + body + jaw + eyes + hands (axis-angle, radians).
    - transl uses Th (world translation).
    - Optional world Rh (3x3) applied after SMPL-X vertices (dataset sets Rh=I).
    """
    import torch

    # Split pose165 (axis-angle blocks)
    go = torch.as_tensor(pose165[:3], dtype=torch.float32)[None]
    body = torch.as_tensor(pose165[3:3+63], dtype=torch.float32)[None]
    jaw = torch.as_tensor(pose165[3+63:3+63+3], dtype=torch.float32)[None]
    leye = torch.as_tensor(pose165[3+63+3:3+63+3+3], dtype=torch.float32)[None]
    reye = torch.as_tensor(pose165[3+63+3+3:3+63+3+3+3], dtype=torch.float32)[None]
    hands = torch.as_tensor(pose165[3+63+3+3+3:], dtype=torch.float32)
    lhand = hands[:45][None]
    rhand = hands[45:][None]

    kwargs = dict(
        global_orient=go,
        body_pose=body,
        jaw_pose=jaw,
        leye_pose=leye,
        reye_pose=reye,
        left_hand_pose=lhand,
        right_hand_pose=rhand,
    )
    if beta is not None:
        kwargs['betas'] = torch.as_tensor(beta, dtype=torch.float32)[None]
    if Th is not None:
        kwargs['transl'] = torch.as_tensor(Th, dtype=torch.float32)[None]

    out = smplx_model(**kwargs)
    verts = out.vertices.detach().cpu().numpy()[0]
    if Rh is not None:
        verts = (Rh @ verts.T).T
    if sample_points and sample_points > 0:
        faces = np.asarray(getattr(smplx_model, 'faces'), dtype=np.int64)
        pts = _sample_points_on_mesh(verts.astype(np.float32), faces, int(sample_points))
        return pts
    return verts.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Chamfer(PRED PLY vs SMPL-X world)")
    ap.add_argument('--ply_dir', required=True, help='Directory with frame_*.ply')
    ap.add_argument('--smpl_npz', default='viewer/output/snapshots/ori_smpl_params.npz', help='NPZ with SMPL-X params')
    ap.add_argument('--smpl_model_dir', default='smpl_model/smplx', help='Directory containing SMPLX_NEUTRAL.npz')
    ap.add_argument('--out', default='ROOT/metrics_eval_smplx.json', help='Output metrics JSON path')
    ap.add_argument('--smpl_sample_points', type=int, default=0, help='If >0, uniformly sample this many points on SMPL-X surface')
    ap.add_argument('--max_frames', type=int, default=0, help='Limit number of frames (0=all)')
    args = ap.parse_args()

    ply_dir = Path(args.ply_dir)
    frames = list_frames_from_dir(ply_dir)
    if not frames:
        raise SystemExit(f"No frame_*.ply in {ply_dir}")
    if args.max_frames and len(frames) > args.max_frames:
        frames = frames[:args.max_frames]

    motion = load_npz(Path(args.smpl_npz))
    # Betas broadcasting
    beta = motion.get('beta') or motion.get('betas')
    if beta is not None:
        beta = np.asarray(beta).reshape(-1).astype(np.float32)

    smplx = try_import_smplx()
    model = smplx.SMPLX(
        model_path=str(Path(args.smpl_model_dir)),
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )

    metrics = {"ply_vs_smplx": {}, "summary": {}}
    c2_list, c1_list = [], []

    for frame_idx, ply_path in frames:
        # Pose
        pose165 = compose_pose165_from_npz(motion, frame_idx)
        # World transforms
        Th = select_frame(motion.get('Th') if 'Th' in motion else motion.get('transl'), frame_idx, d=3)
        Rh = select_frame(motion.get('Rh'), frame_idx)
        if Rh is not None:
            Rh = np.asarray(Rh, dtype=np.float32)
            if Rh.shape == (3,):
                # axis-angle to 3x3
                try:
                    from scipy.spatial.transform import Rotation as R
                    Rh = R.from_rotvec(Rh).as_matrix().astype(np.float32)
                except Exception:
                    theta = np.linalg.norm(Rh) + 1e-8
                    if theta < 1e-8:
                        Rh = np.eye(3, dtype=np.float32)
                    else:
                        k = Rh / theta
                        K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], dtype=np.float32)
                        I = np.eye(3, dtype=np.float32)
                        Rh = I + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            elif Rh.shape != (3,3):
                Rh = None

        try:
            ply_pts = load_ply_points_ascii(ply_path)
        except Exception as e:
            metrics['ply_vs_smplx'][str(frame_idx)] = {"error": f"PLY read: {e}"}
            print(f"[{frame_idx}] skip PLY read error: {e}")
            continue

        try:
            smpl_pts = smplx_points_world(model, pose165, beta, Th, Rh, sample_points=args.smpl_sample_points)
        except Exception as e:
            metrics['ply_vs_smplx'][str(frame_idx)] = {"error": f"SMPL-X: {e}"}
            print(f"[{frame_idx}] skip SMPL-X error: {e}")
            continue

        try:
            d = chamfer(ply_pts, smpl_pts)
            metrics['ply_vs_smplx'][str(frame_idx)] = d
            c2_list.append(d['chamfer_L2_cm2'])
            c1_list.append(d['chamfer_L2_cm'])
            print(f"[{frame_idx}] L2_cm2={d['chamfer_L2_cm2']:.4f}, L2_cm={d['chamfer_L2_cm']:.4f}")
        except Exception as e:
            metrics['ply_vs_smplx'][str(frame_idx)] = {"error": f"Chamfer: {e}"}
            print(f"[{frame_idx}] chamfer error: {e}")

    if c2_list:
        arr2 = np.array(c2_list, dtype=np.float32)
        arr1 = np.array(c1_list, dtype=np.float32)
        metrics['summary'] = {
            'count': int(arr2.size),
            'L2_cm2_mean': float(arr2.mean()),
            'L2_cm_mean': float(arr1.mean()),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
