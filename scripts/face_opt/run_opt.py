#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.face_opt.datasets import load_cameras, list_frame_images, save_overlay, undistort_points
from scripts.face_opt.mp_face import detect_face_landmarks, face_is_frontalish, select_face_keypoints_subset
from scripts.face_opt.smplx_utils import build_smplx, load_npz, compose_pose165_from_npz, select_frame, smplx_vertices_world, face_landmarks_3d_from_smplx, coerce_betas_for_model
from scripts.face_opt.optim_head import ViewData, optimize_head_params
from scripts.face_opt.render_compare import smplx_vertices_dataset, project_vertices_undistort


def main():
    ap = argparse.ArgumentParser(description='Multi-view head SMPL-X optimization using MediaPipe face landmarks')
    ap.add_argument('--subject', default='/home/hello/data/SQ_02', help='Subject root containing images/, mattings/, calibration json, smpl_params.npz')
    ap.add_argument('--frame', type=int, default=1800, help='Frame index to optimize (6-digit zero padded)')
    ap.add_argument('--smpl_model_dir', default='smpl_model/smplx', help='Directory containing SMPLX_NEUTRAL.npz')
    ap.add_argument('--out_root', default='output/face_opt', help='Where to write debug + outputs')
    ap.add_argument('--min_face_rel_area', type=float, default=0.05, help='Min relative face box area to keep a view')
    ap.add_argument('--max_views', type=int, default=12, help='Max number of cameras to use')
    ap.add_argument('--max_iter', type=int, default=100, help='Optimizer iterations')
    # Landmark mapping options
    ap.add_argument('--flame_map', type=str, default='', help='Path to FLAME->SMPL-X vertex map npy (e.g., SMPL-X__FLAME_vertex_ids.npy)')
    ap.add_argument('--flame_lmk_idx', type=str, default='', help='Path to FLAME landmark vertex indices (npy/txt/csv). If provided with --mp_lmk_idx, will override default 3D landmark selection')
    ap.add_argument('--mp_lmk_idx', type=str, default='', help='MediaPipe landmark indices (txt/csv with integers or comma-separated string) to match FLAME landmarks length')
    ap.add_argument('--mp_embed', type=str, default='/home/hello/code/mmlphuman/assets/mediapipe_landmark_embedding.npz', help='Path to MediaPipe-SMPLX barycentric embedding npz (e.g., assets/mediapipe_landmark_embedding.npz)')
    # Render compare (dataset-style projection) options
    ap.add_argument('--render_compare', action='store_true', help='After optimization, render original vs optimized projections using dataset-specific pipeline')
    ap.add_argument('--render_cams', type=str, default='', help='Comma-separated camera IDs to render; default uses selected cameras')
    ap.add_argument('--compare_out_root', type=str, default='output/face_opt_compare', help='Output root for compare renders')
    args = ap.parse_args()

    subject_dir = Path(args.subject)
    out_dir = Path(args.out_root) / subject_dir.name / f"{args.frame:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cameras and frame images
    cams = load_cameras(subject_dir)
    frame_imgs = list_frame_images(subject_dir, args.frame)
    if not frame_imgs:
        raise SystemExit(f"No images found for frame {args.frame} in {subject_dir}")

    # Detect faces and select views
    selected: List[Dict] = []
    import cv2
    # Prefer using 105 MediaPipe landmarks if embedding indices are available
    mp_idx_for_2d = None
    try:
        mp_embed_default = Path(args.mp_embed) if args.mp_embed else Path('assets/mediapipe_landmark_embedding.npz')
        if mp_embed_default.is_file():
            data = np.load(str(mp_embed_default), allow_pickle=True)
            if 'landmark_indices' in data:
                mp_idx_for_2d = np.asarray(data['landmark_indices'], dtype=np.int64).reshape(-1)
    except Exception:
        mp_idx_for_2d = None
    for cam_id, img_path in frame_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = detect_face_landmarks(img)
        if not faces:
            continue
        f = faces[0]
        if not face_is_frontalish(f, img.shape, min_rel_size=args.min_face_rel_area):
            continue
        # 使用105点进行优化（如果可用），同时也保存全部468关键点便于可视检查
        if mp_idx_for_2d is not None and mp_idx_for_2d.size > 0:
            sub = f.keypoints[mp_idx_for_2d]
            idx = mp_idx_for_2d.tolist()
        else:
            # 回退到稳定的5点子集（旧行为）
            print(f"[WARN] MediaPipe embedding indices not found")
            break
        # 保存全部人脸关键点可视化（468点）
        save_overlay(out_dir / f"{cam_id}_face_lmk_all.png", img, f.keypoints)
        # 保存用于优化的关键点（105）
        save_overlay(out_dir / f"{cam_id}_face_lmk.png", img, sub)
        selected.append({
            'cam_id': cam_id,
            'img': str(img_path),
            'kpts2d': sub.astype(np.float32).tolist(),
            'kpts2d_all': f.keypoints.astype(np.float32).tolist(),
            'kpts2d_all_count': int(f.keypoints.shape[0]),
            'kpts_idx': idx,
            'bbox': f.bbox,
        })
        if len(selected) >= args.max_views:
            break

    # Save selection summary
    with open(out_dir / 'selected_cameras.json', 'w', encoding='utf-8') as f:
        json.dump({'frame': args.frame, 'views': selected}, f, indent=2)
    if len(selected) == 0:
        raise SystemExit('No views with detectable frontal-ish faces')

    # Load SMPL-X model + motion
    smplx_model = build_smplx(Path(args.smpl_model_dir))
    motion = load_npz(subject_dir / 'smpl_params.npz')
    pose165 = compose_pose165_from_npz(motion, args.frame)
    Th = select_frame(motion.get('Th') if 'Th' in motion else motion.get('transl'), args.frame, d=3)
    if Th is None:
        Th = np.zeros(3, dtype=np.float32)
    # Prefer 'betas'; fallback to 'beta'. Then coerce to model's expected length.
    beta_raw = motion.get('betas')
    if beta_raw is None:
        beta_raw = motion.get('beta')
    beta = coerce_betas_for_model(beta_raw, smplx_model.model)

    # Build initial 3D landmarks from SMPL-X (dataset-style verts for correctness)
    Rh_world = select_frame(motion.get('Rh'), args.frame)
    verts0 = smplx_vertices_dataset(smplx_model.model, pose165, beta, Th, Rh_world)

    def _load_idx_list(txt: str) -> np.ndarray:
        p = Path(txt)
        if p.suffix.lower() == '.npy' and p.is_file():
            arr = np.load(p, allow_pickle=True)
            return np.asarray(arr, dtype=np.int64).reshape(-1)
        # parse csv/list
        try:
            if p.is_file():
                raw = p.read_text()
            else:
                raw = txt
            items = [s for s in raw.replace('\n', ',').replace('\t', ',').split(',') if s.strip()!='']
            return np.asarray([int(s.strip()) for s in items], dtype=np.int64)
        except Exception:
            return np.zeros((0,), dtype=np.int64)

    # Default: use built-in small subset
    face3d, lmk3d_ids = face_landmarks_3d_from_smplx(verts0, smplx_model.faces)
    lmk_face_idx = None
    lmk_bary = None
    use_mp_embed = False
    # If FLAME mapping provided with index list, override 3D landmarks
    flame_map_path = Path(args.flame_map) if args.flame_map else None
    flame_idx_path = Path(args.flame_lmk_idx) if args.flame_lmk_idx else None
    mp_idx_list = _load_idx_list(args.mp_lmk_idx) if args.mp_lmk_idx else np.zeros((0,), dtype=np.int64)
    use_flame = flame_map_path and flame_map_path.is_file() and flame_idx_path and (flame_idx_path.is_file() or args.flame_lmk_idx)
    if use_flame:
        from scripts.face_opt.smplx_utils import load_flame_to_smplx_vertex_map, face_landmarks_from_flame_map
        flame2smplx = load_flame_to_smplx_vertex_map(flame_map_path)
        flame_lmk_idx = _load_idx_list(args.flame_lmk_idx)
        if flame_lmk_idx.size > 0:
            face3d, lmk3d_ids = face_landmarks_from_flame_map(verts0, flame2smplx, flame_lmk_idx)
            print(f"Use FLAME landmarks: {flame_lmk_idx.size} points -> SMPL-X indices {len(lmk3d_ids)}")
        else:
            print("FLAME map provided but no valid flame_lmk_idx; fallback to default subset")
    else:
        # If MediaPipe embedding is provided, use its barycentric mapping (preferred when available)
        mp_embed_path = Path(args.mp_embed) if args.mp_embed else Path('assets/mediapipe_landmark_embedding.npz')
        if mp_embed_path.is_file():
            from scripts.face_opt.smplx_utils import load_mediapipe_embedding, points_from_barycentric
            lmk_face_idx, lmk_bary, mp_idx_emb = load_mediapipe_embedding(mp_embed_path)
            face3d = points_from_barycentric(verts0, smplx_model.faces, lmk_face_idx, lmk_bary)
            lmk3d_ids = None
            if mp_idx_emb is not None and mp_idx_list.size == 0:
                mp_idx_list = mp_idx_emb
            use_mp_embed = True
            print(f"Use MediaPipe embedding: {face3d.shape[0]} landmarks from {mp_embed_path}")

    # Prepare views for optimization
    views: List[ViewData] = []
    for v in selected:
        cam_id = v['cam_id']
        cam = cams[cam_id]
        # Build 2D target
        if use_mp_embed and 'kpts2d_all' in v and v.get('kpts2d_all_count', 0) > 0:
            all2d = np.array(v['kpts2d_all'], dtype=np.float32)
            target2d = all2d[mp_idx_list] if mp_idx_list.size > 0 else all2d[:face3d.shape[0]]
        else:
            target2d = np.array(v['kpts2d'], dtype=np.float32)
        target2d_ud = undistort_points(target2d, cam.K, cam.dist)
        dist = cam.dist.astype(np.float32) if cam.dist is not None else np.zeros(5, dtype=np.float32)
        views.append(ViewData(cam_id=cam_id, K=cam.K.astype(np.float32), R=cam.R.astype(np.float32), T=cam.T.astype(np.float32), dist=dist, target_2d=target2d, lmk_idx_desc=v['kpts_idx'], target_2d_ud=target2d_ud))

    # Calibrate extrinsics direction per camera using initial landmarks; save initial overlay
    def _reproj_err(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, T: np.ndarray, tgt_ud: np.ndarray) -> float:
        Xc = (R @ pts3d.T) + T.reshape(3, 1)
        Z = Xc[2, :] + 1e-8
        uv = (K @ Xc) / Z
        uv = uv[:2, :].T
        d = uv - tgt_ud
        return float(np.mean(np.sum(d * d, axis=1)))

    # Initial 3D face points
    if lmk_face_idx is not None and lmk_bary is not None:
        from scripts.face_opt.smplx_utils import points_from_barycentric
        face3d_0 = points_from_barycentric(verts0, smplx_model.faces, lmk_face_idx, lmk_bary)
    else:
        face3d_0 = verts0[lmk3d_ids, :]

    for v in views:
        import cv2
        # Test R,T vs R^T,-R^T T
        R_a, T_a = v.R.astype(np.float32), v.T.astype(np.float32)
        R_b = R_a.T
        T_b = (-R_a.T @ T_a.reshape(3)).reshape(3)
        err_a = _reproj_err(face3d_0.astype(np.float32), v.K.astype(np.float32), R_a, T_a.reshape(3), v.target_2d_ud.astype(np.float32))
        err_b = _reproj_err(face3d_0.astype(np.float32), v.K.astype(np.float32), R_b, T_b.reshape(3), v.target_2d_ud.astype(np.float32))
        if err_b < err_a:
            v.R, v.T = R_b, T_b
            print(f"[calib] cam {v.cam_id}: use R^T,-R^T T (errB={err_b:.2f} < errA={err_a:.2f})")
        else:
            print(f"[calib] cam {v.cam_id}: use R,T (errA={err_a:.2f} <= errB={err_b:.2f})")
        # Save initial overlay
        img_path = next((p for cid, p in frame_imgs if cid == v.cam_id), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        img_ud = cv2.undistort(img, v.K.astype(np.float32), v.dist.astype(np.float32))
        Xc = (v.R.astype(np.float32) @ face3d_0.astype(np.float32).T) + v.T.astype(np.float32).reshape(3, 1)
        Z = Xc[2, :] + 1e-8
        pts2d0 = (v.K.astype(np.float32) @ Xc) / Z
        pts2d0 = pts2d0[:2, :].T
        vis0 = img_ud.copy()
        for (x, y) in v.target_2d_ud.astype(np.int32):
            cv2.circle(vis0, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        for (x, y) in pts2d0.astype(np.int32):
            cv2.circle(vis0, (int(x), int(y)), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{v.cam_id}_reproj_init_vs_det.png"), vis0)
        save_overlay(out_dir / f"{v.cam_id}_reproj_init.png", img_ud, pts2d0, color=(0, 255, 255))

    # Optimize head params (fix betas)
    # Use world Rh if available from dataset
    Rh_world = select_frame(motion.get('Rh'), args.frame)
    pose_new, Th_new = optimize_head_params(views, smplx_model.model, beta, pose165, Th,
                                            lmk3d_ids.tolist() if lmk3d_ids is not None else None,
                                            lmk_face_idx=lmk_face_idx, lmk_bary=lmk_bary,
                                            Rh_world=Rh_world,
                                            max_iter=args.max_iter)

    # Save outputs
    np.savez(out_dir / 'smplx_optimized.npz', pose=pose_new, Th=Th_new, betas=beta)
    print(f"Optimized pose/Th saved to {out_dir}/smplx_optimized.npz")

    # Project and visualize reprojection after optimization (dataset-style vertices)
    Rh_world = select_frame(motion.get('Rh'), args.frame)
    verts1 = smplx_vertices_dataset(smplx_model.model, pose_new, beta, Th_new, Rh_world)
    if lmk_face_idx is not None and lmk_bary is not None:
        from scripts.face_opt.smplx_utils import points_from_barycentric
        face3d_1 = points_from_barycentric(verts1, smplx_model.faces, lmk_face_idx, lmk_bary)
    else:
        face3d_1 = verts1[lmk3d_ids, :]
    for v in views:
        import cv2
        # Find original image path for this cam_id
        try:
            img_path = next(p for cid, p in frame_imgs if cid == v.cam_id)
        except StopIteration:
            # Should not happen; skip if no image found
            continue
        img = cv2.imread(str(img_path))
        # Undistort for visualization to match dataset projection
        img_ud = cv2.undistort(img, v.K.astype(np.float32), v.dist.astype(np.float32))
        rvec, _ = cv2.Rodrigues(v.R)
        tvec = v.T.reshape(3, 1)
        # Dataset-style projection (no distortion on undistorted image)
        Xc = (v.R.astype(np.float32) @ face3d_1.astype(np.float32).T) + v.T.astype(np.float32).reshape(3, 1)
        Z = Xc[2, :] + 1e-8
        pts2d = (v.K.astype(np.float32) @ Xc) / Z
        pts2d = pts2d[:2, :].T
        # Debug check for counts
        if pts2d.shape[0] != v.target_2d.shape[0]:
            print(f"[WARN] points count mismatch cam {v.cam_id}: reproj={pts2d.shape[0]} det={v.target_2d.shape[0]}")
        # Also draw detected 2D targets (green) alongside reprojection (red)
        vis = img_ud.copy()
        # Draw detected undistorted (target) in green
        for (x, y) in v.target_2d_ud.astype(np.int32):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        # Draw reprojected in red
        for (x, y) in pts2d.astype(np.int32):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        # Save combined visualization for direct comparison
        out_comb = out_dir / f"{v.cam_id}_reproj_vs_det.png"
        cv2.imwrite(str(out_comb), vis)
        # Keep red-only image for reference（相同点数）
        save_overlay(out_dir / f"{v.cam_id}_reproj_opt.png", img_ud, pts2d, color=(0, 0, 255))

    # Optional: render dataset-style compare (original vs optimized) per camera
    if args.render_compare:
        import cv2
        # Load original motion npz
        orig_npz_path = subject_dir / 'smpl_params.npz'
        motion_orig = load_npz(orig_npz_path)
        pose_orig = compose_pose165_from_npz(motion_orig, args.frame)
        Th_orig = select_frame(motion_orig.get('Th') if 'Th' in motion_orig else motion_orig.get('transl'), args.frame, d=3)
        Rh_orig = select_frame(motion_orig.get('Rh'), args.frame)
        beta_orig = coerce_betas_for_model(motion_orig.get('betas') if 'betas' in motion_orig else motion_orig.get('beta'), smplx_model.model)

        # Use orig Rh if optimized doesn't include it
        Rh_new = Rh_orig
        # Compute vertices with dataset transforms
        V_orig = smplx_vertices_dataset(smplx_model.model, pose_orig, beta_orig, Th_orig, Rh_orig)
        V_opt = smplx_vertices_dataset(smplx_model.model, pose_new, beta, Th_new, Rh_new)

        # Decide cameras to render
        if args.render_cams:
            cam_ids = [c.strip() for c in args.render_cams.split(',') if c.strip() in cams]
        else:
            cam_ids = [v.cam_id for v in views] if views else []
            if not cam_ids and cams:
                cam_ids = [sorted(cams.keys())[0]]

        cmp_dir = Path(args.compare_out_root) / subject_dir.name / f"{args.frame:06d}"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        for cid in cam_ids:
            cam = cams[cid]
            img_path = subject_dir / 'images' / cid / f"{args.frame:06d}.jpg"
            img0 = cv2.imread(str(img_path))
            if img0 is None:
                continue
            img_ud, uv_o = project_vertices_undistort(img0, V_orig, cam.K, cam.R, cam.T, cam.dist)
            _, uv_n = project_vertices_undistort(img0, V_opt, cam.K, cam.R, cam.T, cam.dist)
            vis = img_ud.copy()
            # Draw original in blue, optimized in red
            for (x, y) in uv_o:
                cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            for (x, y) in uv_n:
                cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            out_path = cmp_dir / f"{cid}_compare.png"
            cv2.imwrite(str(out_path), vis)
            print(f"Saved compare: {out_path}")


if __name__ == '__main__':
    main()
