#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate binary mask IoU and Chamfer distance for PLY pairs with matching names.

Usage:
  python scripts/eval_masks_and_ply.py \
      --root <root_dir> \
      [--mask_dir mask --mask_render_dir mask_render] \
      [--ply_a ply_a --ply_b ply_b] \
      [--mask_thresh 127] [--sample_points 100000] \
      [--out metrics_eval.json]

Assumptions:
  - <root_dir>/mask and <root_dir>/mask_render contain images with matching names.
  - <root_dir>/<ply_a> and <root_dir>/<ply_b> contain PLYs with matching names.
  - PLYs can be point clouds or triangle meshes (meshes will be uniformly sampled).

Outputs:
  - Prints per-file metrics and summary.
  - Writes a metrics JSON at --out (default: <root_dir>/metrics_eval.json).
"""

import argparse
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def load_mask(path: Path, thresh: int = 127) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return arr > thresh


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        # center-crop to min common shape as fallback
        H = min(mask_a.shape[0], mask_b.shape[0])
        W = min(mask_a.shape[1], mask_b.shape[1])
        mask_a = mask_a[:H, :W]
        mask_b = mask_b[:H, :W]
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def find_common_files(dir_a: Path, dir_b: Path, exts: Tuple[str, ...]) -> List[str]:
    a_files = {p.stem for p in dir_a.iterdir() if p.is_file() and p.suffix.lower() in exts}
    b_files = {p.stem for p in dir_b.iterdir() if p.is_file() and p.suffix.lower() in exts}
    commons = sorted(a_files & b_files)
    return commons


def try_import_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except Exception:
        return None


def try_import_trimesh():
    try:
        import trimesh  # type: ignore
        return trimesh
    except Exception:
        return None


def load_ply_points(ply_path: Path, sample_points: int = 100000) -> np.ndarray:
    """Load PLY as point cloud; if mesh, uniformly sample points on surface.
    Returns Nx3 float32 array.
    """
    o3d = try_import_open3d()
    if o3d is not None:
        # First try as point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd is not None and len(pcd.points) > 0:
            pts = np.asarray(pcd.points, dtype=np.float32)
            return pts
        # Fallback: try as mesh and sample
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if mesh is not None and len(mesh.triangles) > 0:
            mesh.compute_triangle_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
            pts = np.asarray(pcd.points, dtype=np.float32)
            return pts

    # Fallback to trimesh if open3d not available
    tm = try_import_trimesh()
    if tm is not None:
        m = tm.load(str(ply_path), process=False)
        # Point cloud
        if isinstance(m, tm.points.PointCloud):
            pts = np.asarray(m.vertices, dtype=np.float32)
            return pts
        # Mesh
        if isinstance(m, tm.Trimesh):
            # Sample points on surface
            n = int(sample_points)
            pts, _ = tm.sample.sample_surface_even(m, n)
            return np.asarray(pts, dtype=np.float32)
        # Scene containing geometry
        if isinstance(m, tm.Scene) and hasattr(m, 'geometry') and len(m.geometry):
            # Combine geometry by sampling from each mesh
            pts_all = []
            for g in m.geometry.values():
                if isinstance(g, tm.Trimesh):
                    k = max(1000, int(sample_points // len(m.geometry)))
                    pts, _ = tm.sample.sample_surface_even(g, k)
                    pts_all.append(np.asarray(pts, dtype=np.float32))
            if pts_all:
                return np.concatenate(pts_all, axis=0)

    # Last resort: simple ASCII PLY reader for vertex-only files
    try:
        with open(ply_path, 'rb') as f:
            header = []
            while True:
                line = f.readline()
                header.append(line)
                if line.strip() == b'end_header':
                    break
            header_txt = b''.join(header).decode('utf-8', errors='ignore')
            if 'format ascii' not in header_txt:
                raise RuntimeError('Binary PLY without open3d/trimesh support')
        # Load ASCII via numpy
        data = np.loadtxt(ply_path, dtype=np.float32, comments=['ply', 'format', 'element', 'property', 'comment', 'obj_info', 'end_header'])
        # Heuristic: first 3 columns are x,y,z
        if data.ndim == 1:
            data = data[None, :]
        pts = data[:, :3].astype(np.float32)
        return pts
    except Exception as e:
        raise RuntimeError(f"Cannot load PLY: {ply_path} ({e})")


def chamfer_distances(pts_a: np.ndarray, pts_b: np.ndarray) -> Dict[str, float]:
    """Compute symmetric Chamfer-L2 (cm^2) and Chamfer-L1 (cm) using SciPy KDTree."""
    from scipy.spatial import cKDTree
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return {"chamfer_L2_cm2": float('nan'), "chamfer_L1_cm": float('nan')}
    # meters -> cm
    tree_b = cKDTree(pts_b)
    d_a2b, _ = tree_b.query(pts_a, k=1, workers=-1)
    tree_a = cKDTree(pts_a)
    d_b2a, _ = tree_a.query(pts_b, k=1, workers=-1)
    d_cm_a2b = d_a2b * 100.0
    d_cm_b2a = d_b2a * 100.0
    c2 = 0.5 * (np.mean(d_cm_a2b**2) + np.mean(d_cm_b2a**2))
    c1 = 0.5 * (np.mean(d_cm_a2b) + np.mean(d_cm_b2a))
    return {"chamfer_L2_cm2": float(c2), "chamfer_L1_cm": float(c1)}


def parse_frame_range(txt: str) -> List[int]:
    txt = txt.strip().replace('-', ':')
    parts = [p for p in txt.split(':') if p != '']
    if not parts:
        return []
    try:
        if len(parts) == 1:
            i = int(parts[0]); return [i]
        elif len(parts) == 2:
            a, b = int(parts[0]), int(parts[1])
            if b < a:
                a, b = b, a
            return list(range(a, b + 1))
        else:
            a, b, s = int(parts[0]), int(parts[1]), int(parts[2])
            if s == 0:
                s = 1
            if b < a:
                a, b = b, a
            return list(range(a, b + 1, s))
    except ValueError:
        return []


def find_ply_b_for_frame(bdir: Path, frame: int, template: Optional[str] = None) -> Optional[Path]:
    if template:
        try:
            name = template.format(frame=frame)
            p = bdir / name
            return p if p.is_file() else None
        except Exception:
            pass
    candidates: List[Path] = []
    candidates.append(bdir / f"{frame}.ply")
    for z in (4, 5, 6, 8):
        candidates.append(bdir / f"{frame:0{z}d}.ply")
        candidates.append(bdir / f"frame_{frame:0{z}d}.ply")
    candidates.append(bdir / f"frame_{frame}.ply")
    for p in candidates:
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Evaluate IoU for masks and Chamfer for PLYs")
    ap.add_argument('--root', required=True, help='Root directory')
    ap.add_argument('--mask_dir', default='mask', help='Subdir under root for GT masks')
    ap.add_argument('--mask_render_dir', default='mask_render', help='Subdir under root for predicted masks')
    ap.add_argument('--mask_thresh', type=int, default=127, help='Threshold for binarizing grayscale masks')
    ap.add_argument('--ply_a', default=None, help='Subdir under root for PLY set A (e.g., GT)')
    ap.add_argument('--ply_b', default=None, help='Subdir under root or absolute path for PLY set B (e.g., Pred)')
    # Per-frame A vs single-dir B mode
    ap.add_argument('--ply_frames_root', default=None, help='Root where subfolders are named by frame numbers (e.g., 1701/1702/...)')
    ap.add_argument('--ply_frames_rel', default='dense/fuse.ply', help='Relative path inside each frame folder (default: dense/fuse.ply)')
    ap.add_argument('--ply_frames_range', default=None, help="Frame range like '1701-2000' or '1701:2000[:step]'")
    ap.add_argument('--ply_b_template', default=None, help="Filename template for B, e.g., 'frame_{frame:05d}.ply'")
    ap.add_argument('--sample_points', type=int, default=100000, help='Num points to sample from mesh for Chamfer')
    ap.add_argument('--out', default=None, help='Output metrics JSON path (default: <root>/metrics_eval.json)')
    args = ap.parse_args()

    root = Path(args.root)
    mask_dir = root / args.mask_dir
    mask_render_dir = root / args.mask_render_dir

    metrics = {"mask_iou": {}, "mask_iou_summary": {}, "ply_chamfer": {}, "ply_chamfer_summary": {}}

    # 1) IoU for masks
    if mask_dir.is_dir() and mask_render_dir.is_dir():
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        commons = find_common_files(mask_dir, mask_render_dir, exts)
        ious = []
        for stem in commons:
            a_path = next((mask_dir / f"{stem}{ext}" for ext in exts if (mask_dir / f"{stem}{ext}").exists()), None)
            b_path = next((mask_render_dir / f"{stem}{ext}" for ext in exts if (mask_render_dir / f"{stem}{ext}").exists()), None)
            if a_path is None or b_path is None:
                continue
            m1 = load_mask(a_path, args.mask_thresh)
            m2 = load_mask(b_path, args.mask_thresh)
            iou = compute_iou(m1, m2)
            metrics["mask_iou"][stem] = iou
            ious.append(iou)
        if ious:
            arr = np.array(ious, dtype=np.float32)
            metrics["mask_iou_summary"] = {
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }
        print(f"Mask IoU pairs: {len(ious)}")
        if ious:
            print(f"Mask IoU mean: {metrics['mask_iou_summary']['mean']:.4f}")
    else:
        print(f"Skip mask IoU: missing '{mask_dir}' or '{mask_render_dir}'")

    # 2) Chamfer for PLYs
    did_chamfer = False
    if args.ply_frames_root and args.ply_frames_range and args.ply_b:
        frames = parse_frame_range(args.ply_frames_range)
        a_root = Path(args.ply_frames_root)
        b_dir = Path(args.ply_b)
        if not a_root.is_absolute():
            a_root = root / a_root
        if not b_dir.is_absolute():
            b_dir = root / b_dir
        c2_list, c1_list = [], []
        print(f"Chamfer per-frame: {len(frames)} frames; A={a_root}, rel='{args.ply_frames_rel}', B={b_dir}")
        for frame in frames:
            pa = a_root / str(frame) / args.ply_frames_rel
            pb = find_ply_b_for_frame(b_dir, frame, template=args.ply_b_template)
            key = str(frame)
            if not pa.is_file():
                metrics.setdefault("ply_chamfer", {})[key] = {"error": f"missing {pa}"}
                print(f"  {key}: missing A {pa}")
                continue
            if pb is None:
                metrics.setdefault("ply_chamfer", {})[key] = {"error": f"missing in {b_dir}"}
                print(f"  {key}: missing B in {b_dir}")
                continue
            try:
                a_pts = load_ply_points(pa, sample_points=args.sample_points)
                b_pts = load_ply_points(pb, sample_points=args.sample_points)
                d = chamfer_distances(a_pts, b_pts)
                metrics["ply_chamfer"][key] = d
                c2_list.append(d["chamfer_L2_cm2"])
                c1_list.append(d["chamfer_L1_cm"])
                print(f"  {key}: L2_cm2={d['chamfer_L2_cm2']:.4f}, L1_cm={d['chamfer_L1_cm']:.4f}")
            except Exception as e:
                metrics["ply_chamfer"][key] = {"error": str(e)}
                print(f"  {key}: error {e}")
        if c2_list:
            c2 = np.array(c2_list, dtype=np.float32)
            c1 = np.array(c1_list, dtype=np.float32)
            metrics["ply_chamfer_summary"] = {
                "count": int(c2.size),
                "L2_cm2_mean": float(c2.mean()),
                "L1_cm_mean": float(c1.mean()),
            }
            print(f"Chamfer L2_cm2 mean: {metrics['ply_chamfer_summary']['L2_cm2_mean']:.4f}")
        did_chamfer = True

    if not did_chamfer:
        if args.ply_a and args.ply_b:
            ply_a_dir = root / args.ply_a
            ply_b_dir = root / args.ply_b
        else:
            candidates = [p for p in root.iterdir() if p.is_dir() and p.name.lower() not in {args.mask_dir.lower(), args.mask_render_dir.lower()}]
            ply_dirs = []
            for p in candidates:
                if any(c.suffix.lower() == '.ply' for c in p.iterdir() if c.is_file()):
                    ply_dirs.append(p)
            if len(ply_dirs) >= 2:
                ply_a_dir, ply_b_dir = ply_dirs[:2]
            else:
                ply_a_dir = ply_b_dir = None

        if ply_a_dir and ply_b_dir and ply_a_dir.is_dir() and ply_b_dir.is_dir():
            commons = find_common_files(ply_a_dir, ply_b_dir, exts=('.ply',))
            c2_list, c1_list = [], []
            print(f"Chamfer pairs: {len(commons)} (A: {ply_a_dir.name}, B: {ply_b_dir.name})")
            for stem in commons:
                pa = ply_a_dir / f"{stem}.ply"
                pb = ply_b_dir / f"{stem}.ply"
                try:
                    a_pts = load_ply_points(pa, sample_points=args.sample_points)
                    b_pts = load_ply_points(pb, sample_points=args.sample_points)
                    d = chamfer_distances(a_pts, b_pts)
                    metrics["ply_chamfer"][stem] = d
                    c2_list.append(d["chamfer_L2_cm2"])
                    c1_list.append(d["chamfer_L1_cm"])
                    print(f"  {stem}: L2_cm2={d['chamfer_L2_cm2']:.4f}, L1_cm={d['chamfer_L1_cm']:.4f}")
                except Exception as e:
                    metrics["ply_chamfer"][stem] = {"error": str(e)}
                    print(f"  {stem}: error {e}")
            if c2_list:
                c2 = np.array(c2_list, dtype=np.float32)
                c1 = np.array(c1_list, dtype=np.float32)
                metrics["ply_chamfer_summary"] = {
                    "count": int(c2.size),
                    "L2_cm2_mean": float(c2.mean()),
                    "L1_cm_mean": float(c1.mean()),
                }
                print(f"Chamfer L2_cm2 mean: {metrics['ply_chamfer_summary']['L2_cm2_mean']:.4f}")
        else:
            print("Skip Chamfer: PLY directories not provided or not found.")

    out_path = Path(args.out) if args.out else (root / 'metrics_eval.json')
    ensure_dir(out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == '__main__':
    main()
