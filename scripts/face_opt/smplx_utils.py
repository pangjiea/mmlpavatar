from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np


def try_import_smplx():
    try:
        import smplx  # type: ignore
        return smplx
    except Exception as e:
        raise RuntimeError(f"smplx not available: {e}")


def try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        raise RuntimeError(f"torch not available: {e}")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compose_pose165_from_npz(motion: Dict[str, np.ndarray], idx: int) -> np.ndarray:
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
        if arr.ndim == 1:
            if arr.shape[0] != d:
                if arr.shape[0] > d:
                    arr = arr[:d]
                else:
                    arr = np.pad(arr, (0, d-arr.shape[0]))
            return arr.astype(np.float32)
        i = min(idx, arr.shape[0]-1)
        out = arr[i]
        if out.shape[0] != d:
            if out.shape[0] > d:
                out = out[:d]
            else:
                out = np.pad(out, (0, d-out.shape[0]))
        return out.astype(np.float32)

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


@dataclass
class SmplxModel:
    model: object
    faces: np.ndarray


def build_smplx(model_dir: Path) -> SmplxModel:
    smplx = try_import_smplx()
    model = smplx.SMPLX(
        model_path=str(model_dir),
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )
    faces = np.asarray(getattr(model, 'faces'), dtype=np.int64)
    return SmplxModel(model=model, faces=faces)


def _infer_num_betas(model) -> int:
    # Try to infer from shapedirs last dimension; fallback to 10.
    nb = 10
    try:
        sd = getattr(model, 'shapedirs', None)
        if sd is not None and hasattr(sd, 'shape'):
            shp = tuple(sd.shape)
            if len(shp) >= 1:
                cand = shp[-1]
                if isinstance(cand, int) and 1 <= cand <= 400:
                    nb = int(cand)
    except Exception:
        pass
    return nb


def coerce_betas_for_model(beta: Optional[np.ndarray], model) -> np.ndarray:
    nb = _infer_num_betas(model)
    if beta is None:
        out = np.zeros(nb, dtype=np.float32)
        return out
    b = np.asarray(beta).reshape(-1).astype(np.float32)
    if b.size != nb:
        # Slice or pad to expected length
        if b.size > nb:
            b = b[:nb]
        else:
            b = np.pad(b, (0, nb - b.size))
    return b.astype(np.float32)


def smplx_vertices_world(smplx_model: SmplxModel, pose165: np.ndarray, beta: Optional[np.ndarray], Th: Optional[np.ndarray]) -> np.ndarray:
    torch = try_import_torch()
    model = smplx_model.model

    # Ensure betas length matches model
    beta = coerce_betas_for_model(beta, model)

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

    out = model(**kwargs)
    verts = out.vertices.detach().cpu().numpy()[0]
    return verts.astype(np.float32)


def face_landmarks_3d_from_smplx(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate 3D facial landmarks from SMPL-X mesh using a fixed subset of indices.

    This is a fallback solution if full FLAME 68-landmark extraction is unavailable.
    The indices approximate: eye corners, nose tip, mouth corners, chin.
    Returns (points_3d, indices_used).
    """
    # NOTE: These indices are approximate and may require calibration per model version.
    # They were chosen to lie in typical facial regions on SMPL-X topology.
    # Order must match mp_face.select_face_keypoints_subset():
    # [right eye outer, left eye outer, nose tip, mouth right, mouth left]
    approx_ids = [
        4540,  # right eye outer approx
        9120,  # left eye outer approx
        3388,  # nose tip approx
        6870,  # mouth right corner approx
        6570,  # mouth left corner approx
    ]
    approx_ids = [i for i in approx_ids if 0 <= i < verts.shape[0]]
    pts = verts[np.array(approx_ids, dtype=np.int64)]
    return pts.astype(np.float32), np.array(approx_ids, dtype=np.int64)


def load_flame_to_smplx_vertex_map(path: Path) -> np.ndarray:
    """Load FLAME->SMPL-X vertex map (length ~5023). Each value is an index in SMPL-X verts."""
    arr = np.load(str(path), allow_pickle=True)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    return arr


def face_landmarks_from_flame_map(verts_smplx: np.ndarray, flame2smplx: np.ndarray, flame_lmk_idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Map FLAME landmark vertex indices to SMPL-X vertices via provided mapping array."""
    flame_lmk_idx = np.asarray(list(flame_lmk_idx), dtype=np.int64).reshape(-1)
    smplx_idx = flame2smplx[flame_lmk_idx]
    pts = verts_smplx[smplx_idx]
    return pts.astype(np.float32), smplx_idx.astype(np.int64)


def load_mediapipe_embedding(path: Path):
    """Load MediaPipe landmark embedding for SMPL-X mesh.

    Expects keys: 'lmk_face_idx' (M,), 'lmk_b_coords' (M,3), optional 'landmark_indices' (M,)
    Returns (face_idx[int64], bary_coords[float32,M,3], mp_indices[int64 or None]).
    """
    data = np.load(str(path), allow_pickle=True)
    face_idx = np.asarray(data['lmk_face_idx'], dtype=np.int64).reshape(-1)
    bcoords = np.asarray(data['lmk_b_coords'], dtype=np.float32).reshape(-1, 3)
    mp_inds = None
    if 'landmark_indices' in data:
        mp_inds = np.asarray(data['landmark_indices'], dtype=np.int64).reshape(-1)
    return face_idx, bcoords, mp_inds


def points_from_barycentric(verts: np.ndarray, faces: np.ndarray, face_idx: np.ndarray, bcoords: np.ndarray) -> np.ndarray:
    """Compute 3D points from barycentric embedding over SMPL-X mesh.

    verts: (V,3), faces: (F,3), face_idx: (M,), bcoords: (M,3)
    """
    tri = faces[face_idx]  # (M,3)
    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]
    w0 = bcoords[:, 0:1]
    w1 = bcoords[:, 1:2]
    w2 = bcoords[:, 2:3]
    pts = w0 * v0 + w1 * v1 + w2 * v2
    return pts.astype(np.float32)
