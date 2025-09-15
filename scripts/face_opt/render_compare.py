#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import cv2

from scripts.face_opt.datasets import load_cameras
from scripts.face_opt.smplx_utils import build_smplx, load_npz, compose_pose165_from_npz, coerce_betas_for_model, select_frame


def axis_angle_to_matrix_np(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32).reshape(-1)
    if r.size == 9:
        return r.reshape(3, 3).astype(np.float32)
    if r.size != 3:
        return np.eye(3, dtype=np.float32)
    theta = np.linalg.norm(r) + 1e-8
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
    I = np.eye(3, dtype=np.float32)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def smplx_vertices_dataset(model, pose165: np.ndarray, betas: np.ndarray, Th: Optional[np.ndarray], Rh: Optional[np.ndarray]) -> np.ndarray:
    import torch
    # Build tensors for model call (no transl; we apply Th/Rh after)
    go = torch.as_tensor(pose165[:3], dtype=torch.float32)[None]
    body = torch.as_tensor(pose165[3:3+63], dtype=torch.float32)[None]
    jaw = torch.as_tensor(pose165[66:69], dtype=torch.float32)[None]
    leye = torch.as_tensor(pose165[69:72], dtype=torch.float32)[None]
    reye = torch.as_tensor(pose165[72:75], dtype=torch.float32)[None]
    lhand = torch.as_tensor(pose165[75:120], dtype=torch.float32)[None]
    rhand = torch.as_tensor(pose165[120:165], dtype=torch.float32)[None]
    kwargs = dict(
        global_orient=go,
        body_pose=body,
        jaw_pose=jaw,
        leye_pose=leye,
        reye_pose=reye,
        left_hand_pose=lhand,
        right_hand_pose=rhand,
    )
    if betas is not None:
        kwargs['betas'] = torch.as_tensor(betas, dtype=torch.float32)[None]
    out = model(**kwargs)
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    # Apply dataset world Rh/Th as in TalkBody4D_utils.py
    Rw = axis_angle_to_matrix_np(Rh) if Rh is not None else np.eye(3, dtype=np.float32)
    Tw = np.asarray(Th, dtype=np.float32).reshape(3) if Th is not None else np.zeros(3, dtype=np.float32)
    verts = (verts @ Rw.T) + Tw[None, :]
    return verts.astype(np.float32)


def project_vertices_undistort(img_bgr: np.ndarray, verts_world: np.ndarray, K: np.ndarray, R: np.ndarray, T: np.ndarray, dist: Optional[np.ndarray]):
    # Undistort image, then project without distortion
    K = K.reshape(3, 3).astype(np.float32)
    R = R.reshape(3, 3).astype(np.float32)
    T = T.reshape(3).astype(np.float32)
    dist = np.asarray(dist, dtype=np.float32).reshape(-1) if dist is not None else np.zeros(5, dtype=np.float32)
    img_ud = cv2.undistort(img_bgr, K, dist)
    Xc = (R @ verts_world.T) + T.reshape(3, 1)  # (3,N)
    Z = Xc[2, :] + 1e-8
    uv = (K @ Xc) / Z
    uv = uv[:2, :].T
    uv_round = np.round(uv).astype(np.int32)
    H, W = img_ud.shape[:2]
    uv_round[:, 0] = np.clip(uv_round[:, 0], 0, W - 1)
    uv_round[:, 1] = np.clip(uv_round[:, 1], 0, H - 1)
    return img_ud, uv_round


def draw_points(img: np.ndarray, pts: np.ndarray, color=(0, 0, 255), radius=1):
    out = img.copy()
    for (x, y) in pts:
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description='Render comparison using dataset projection (TalkBody4D style)')
    ap.add_argument('--subject', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--smpl_model_dir', default='smpl_model/smplx')
    ap.add_argument('--smpl_npz_orig', default=None, help='Original SMPL-X npz (default: <subject>/smpl_params.npz)')
    ap.add_argument('--smpl_npz_opt', required=True, help='Optimized npz from face_opt (pose/Th/betas)')
    ap.add_argument('--out', default=None, help='Output dir (default: output/face_opt_compare/<subject>/<frame:06d>)')
    ap.add_argument('--cams', default='', help='Comma-separated camera IDs to render; default uses selected_cameras.json if available, else first camera.')
    args = ap.parse_args()

    subject_dir = Path(args.subject)
    out_dir = Path(args.out) if args.out else (Path('output/face_opt_compare') / subject_dir.name / f"{args.frame:06d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cameras
    cams = load_cameras(subject_dir)

    # Decide camera list
    cam_list = []
    if args.cams:
        cam_list = [c.strip() for c in args.cams.split(',') if c.strip() in cams]
    else:
        sel_json = Path('output/face_opt') / subject_dir.name / f"{args.frame:06d}" / 'selected_cameras.json'
        if sel_json.is_file():
            import json
            data = json.loads(sel_json.read_text())
            for v in data.get('views', []):
                cid = v.get('cam_id')
                if cid in cams:
                    cam_list.append(cid)
    if not cam_list and cams:
        cam_list = [sorted(cams.keys())[0]]

    # Load SMPL-X model
    smplx = build_smplx(Path(args.smpl_model_dir))

    # Original/Optimized params
    orig_npz = Path(args.smpl_npz_orig) if args.smpl_npz_orig else (subject_dir / 'smpl_params.npz')
    opt_npz = Path(args.smpl_npz_opt)
    orig = load_npz(orig_npz)
    opt = load_npz(opt_npz)

    # Per-frame params (select specific frame for arrays)
    pose_orig = compose_pose165_from_npz(orig, args.frame)
    Th_orig = select_frame(orig.get('Th') if 'Th' in orig else orig.get('transl'), args.frame, d=3)
    Rh_orig = select_frame(orig.get('Rh'), args.frame)
    beta_orig = coerce_betas_for_model(orig.get('betas') if 'betas' in orig else orig.get('beta'), smplx.model)

    pose_opt = compose_pose165_from_npz(opt, args.frame) if 'pose' in opt else pose_orig
    Th_opt = select_frame(opt.get('Th') if 'Th' in opt else opt.get('transl'), args.frame, d=3) if ('Th' in opt or 'transl' in opt) else Th_orig
    Rh_opt = select_frame(opt.get('Rh'), args.frame) if 'Rh' in opt else Rh_orig
    beta_opt = coerce_betas_for_model(opt.get('betas', beta_orig), smplx.model)

    # Compute vertices (dataset transform)
    V_orig = smplx_vertices_dataset(smplx.model, pose_orig, beta_orig, Th_orig, Rh_orig)
    V_opt = smplx_vertices_dataset(smplx.model, pose_opt, beta_opt, Th_opt, Rh_opt)

    # Render per camera
    for cid in cam_list:
        cam = cams[cid]
        img_path = subject_dir / 'images' / cid / f"{args.frame:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        K = cam.K
        R = cam.R if cam.R is not None else np.eye(3, dtype=np.float32)
        T = cam.T if cam.T is not None else np.zeros(3, dtype=np.float32)
        dist = cam.dist

        img_ud, uv_orig = project_vertices_undistort(img, V_orig, K, R, T, dist)
        _, uv_opt = project_vertices_undistort(img, V_opt, K, R, T, dist)

        vis = img_ud.copy()
        # Original in blue, Optimized in red
        for (x, y) in uv_orig:
            cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        for (x, y) in uv_opt:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        out_path = out_dir / f"{cid}_compare.png"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
