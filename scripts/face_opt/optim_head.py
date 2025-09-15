from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scripts.face_opt.smplx_utils import coerce_betas_for_model, points_from_barycentric


@dataclass
class ViewData:
    cam_id: str
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    dist: np.ndarray
    target_2d: np.ndarray  # (M, 2) original image coords
    lmk_idx_desc: List[int]
    target_2d_ud: Optional[np.ndarray] = None  # (M, 2) undistorted coords (aligned with cv2.undistort)


def bundle_loss(params,
                views: List[ViewData],
                smplx_model,
                beta: np.ndarray,
                pose165_init: np.ndarray,
                Th_init: np.ndarray,
                lmk3d_ids: Optional[List[int]] = None,
                lmk_face_idx: Optional[np.ndarray] = None,
                lmk_bary: Optional[np.ndarray] = None,
                Rh_world: Optional[np.ndarray] = None):
    """Compute average L2 loss over views.

    params: concatenated updates for [global_orient(3), jaw(3), leye(3), reye(3), transl(3)]
    We add the delta to the initial values.
    """
    import torch
    import cv2

    device = torch.device('cpu')
    # Unpack
    delta = torch.as_tensor(params, dtype=torch.float32, device=device)
    go = torch.as_tensor(pose165_init[:3], dtype=torch.float32, device=device)
    body = torch.as_tensor(pose165_init[3:3+63], dtype=torch.float32, device=device)
    jaw = torch.as_tensor(pose165_init[66:69], dtype=torch.float32, device=device)
    leye = torch.as_tensor(pose165_init[69:72], dtype=torch.float32, device=device)
    reye = torch.as_tensor(pose165_init[72:75], dtype=torch.float32, device=device)
    lhand = torch.as_tensor(pose165_init[75:120], dtype=torch.float32, device=device)
    rhand = torch.as_tensor(pose165_init[120:165], dtype=torch.float32, device=device)
    Th = torch.as_tensor(Th_init, dtype=torch.float32, device=device)

    # Apply deltas
    go = go + delta[0:3]
    jaw = jaw + delta[3:6]
    leye = leye + delta[6:9]
    reye = reye + delta[9:12]
    Th = Th + delta[12:15]

    smplx = smplx_model
    # Ensure betas length matches model
    beta = coerce_betas_for_model(beta, smplx)
    betas = torch.as_tensor(beta, dtype=torch.float32, device=device)[None]

    # Dataset-style: do NOT pass transl to SMPL-X; apply world Rh/Th after
    out = smplx(
        global_orient=go[None],
        body_pose=body[None],
        jaw_pose=jaw[None],
        leye_pose=leye[None],
        reye_pose=reye[None],
        left_hand_pose=lhand[None],
        right_hand_pose=rhand[None],
        betas=betas,
    )
    verts = out.vertices[0]
    # Apply world transform: verts @ Rh^T + Th
    if Rh_world is not None:
        # Rh_world can be axis-angle(3,) or matrix (3,3)
        Rw = None
        if isinstance(Rh_world, np.ndarray) and Rh_world.shape == (3, 3):
            Rw = Rh_world.astype(np.float32)
        else:
            r = np.asarray(Rh_world, dtype=np.float32).reshape(-1)
            if r.size == 3:
                theta = np.linalg.norm(r) + 1e-8
                if theta < 1e-8:
                    Rw = np.eye(3, dtype=np.float32)
                else:
                    k = r / theta
                    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
                    I = np.eye(3, dtype=np.float32)
                    Rw = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            else:
                Rw = np.eye(3, dtype=np.float32)
        verts = verts @ torch.as_tensor(Rw.T, dtype=torch.float32) + Th[None]
    else:
        # Fallback: add transl to verts if Rh not provided (legacy behavior)
        verts = verts + Th[None]

    # Gather 3D landmark points
    if lmk_face_idx is not None and lmk_bary is not None:
        # barycentric embedding per-iteration (responds to pose updates)
        pts3d = points_from_barycentric(verts.detach().cpu().numpy(), smplx_model.faces, lmk_face_idx, lmk_bary)
    else:
        pts3d = verts[lmk3d_ids, :]
    # Ensure numpy float32 for OpenCV
    if hasattr(pts3d, 'detach'):
        pts3d_np = pts3d.detach().cpu().numpy().astype(np.float32)
    else:
        pts3d_np = np.asarray(pts3d, dtype=np.float32)

    total = 0.0
    count = 0
    for v in views:
        # Dataset projection on undistorted image: uv = (K @ (R*(X)+T))/Z
        K = v.K.astype(np.float32)
        R = v.R.astype(np.float32)
        T = v.T.astype(np.float32).reshape(3)
        Xc = (R @ pts3d_np.T) + T.reshape(3, 1)
        Z = Xc[2, :] + 1e-8
        uv = (K @ Xc) / Z
        uv = uv[:2, :].T.astype(np.float32)
        # Use undistorted target if available
        tgt = v.target_2d_ud if (hasattr(v, 'target_2d_ud') and v.target_2d_ud is not None) else v.target_2d
        diff = uv - tgt.astype(np.float32)
        total += float(np.mean(np.sum(diff * diff, axis=1)))
        count += 1
    return total / max(count, 1)


def optimize_head_params(views: List[ViewData],
                         smplx_model,
                         beta: np.ndarray,
                         pose165_init: np.ndarray,
                         Th_init: np.ndarray,
                         lmk3d_ids: Optional[List[int]] = None,
                         lmk_face_idx: Optional[np.ndarray] = None,
                         lmk_bary: Optional[np.ndarray] = None,
                         Rh_world: Optional[np.ndarray] = None,
                         max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize [go, jaw, leye, reye, Th]. Returns (pose165_new, Th_new)."""
    import numpy as np
    from scipy.optimize import minimize

    x0 = np.zeros(15, dtype=np.float32)
    fun = lambda x: bundle_loss(x, views, smplx_model, beta, pose165_init, Th_init, lmk3d_ids, lmk_face_idx, lmk_bary, Rh_world)
    res = minimize(fun, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-6})
    dx = res.x.astype(np.float32)

    pose_new = pose165_init.copy()
    pose_new[:3] += dx[0:3]
    pose_new[66:69] += dx[3:6]
    pose_new[69:72] += dx[6:9]
    pose_new[72:75] += dx[9:12]
    Th_new = Th_init.copy()
    Th_new += dx[12:15]
    return pose_new, Th_new
