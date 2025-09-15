from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Camera:
    cam_id: str
    K: np.ndarray        # (3, 3)
    R: np.ndarray        # (3, 3)
    T: np.ndarray        # (3,)
    dist: Optional[np.ndarray]  # (5,) or None
    img_size: Tuple[int, int]   # (h, w)

    def world_to_cam(self, xyz_world: np.ndarray) -> np.ndarray:
        # xyz_world: (N, 3)
        # X_cam = R X_world + T
        return (self.R @ xyz_world.T).T + self.T[None, :]

    def project(self, xyz_world: np.ndarray) -> np.ndarray:
        # Project using OpenCV-style pinhole + dist if available
        import cv2
        rvec, _ = cv2.Rodrigues(self.R)
        tvec = self.T.reshape(3, 1)
        K = self.K
        dist = self.dist if self.dist is not None else np.zeros(5, dtype=np.float32)
        pts2d, _ = cv2.projectPoints(xyz_world.astype(np.float32), rvec.astype(np.float32), tvec.astype(np.float32), K.astype(np.float32), dist.astype(np.float32))
        pts2d = pts2d.reshape(-1, 2)
        return pts2d


def load_cameras(subject_dir: Path) -> Dict[str, Camera]:
    calib_full = subject_dir / 'calibration_full.json'
    calib = subject_dir / 'calibration.json'
    meta: Dict[str, dict] = {}
    if calib_full.is_file():
        meta = json.loads(calib_full.read_text())
        cams: Dict[str, Camera] = {}
        for cam_id, c in meta.items():
            K = np.array(c['K'], dtype=np.float64).reshape(3, 3)
            R = np.array(c['R'], dtype=np.float64).reshape(3, 3)
            T = np.array(c['T'], dtype=np.float64).reshape(3)
            img_h, img_w = int(c['imgSize'][0]), int(c['imgSize'][1])
            dist = np.array(c.get('distCoeff', []), dtype=np.float64).reshape(-1) if 'distCoeff' in c else None
            cams[cam_id] = Camera(cam_id=cam_id, K=K, R=R, T=T, dist=dist, img_size=(img_h, img_w))
        return cams
    elif calib.is_file():
        meta = json.loads(calib.read_text())['cameras']
        cams: Dict[str, Camera] = {}
        for cam_id, c in meta.items():
            K = np.array(c['K'], dtype=np.float64)
            R = np.eye(3, dtype=np.float64)
            T = np.zeros(3, dtype=np.float64)
            img_h, img_w = int(c['image_size'][0]), int(c['image_size'][1])
            dist = np.array(c.get('dist', []), dtype=np.float64).reshape(-1) if 'dist' in c else None
            cams[cam_id] = Camera(cam_id=cam_id, K=K, R=R, T=T, dist=dist, img_size=(img_h, img_w))
        return cams
    else:
        raise FileNotFoundError(f"No calibration json in {subject_dir}")


def list_frame_images(subject_dir: Path, frame_idx: int) -> List[Tuple[str, Path]]:
    """Return list of (cam_id, image_path) for cameras that have this frame image.

    Assumes images are in {subject_dir}/images/{cam_id}/{frame:06d}.jpg
    """
    img_root = subject_dir / 'images'
    out: List[Tuple[str, Path]] = []
    fn = f"{frame_idx:06d}.jpg"
    if not img_root.is_dir():
        return out
    for cam_dir in sorted(img_root.iterdir()):
        if cam_dir.is_dir():
            p = cam_dir / fn
            if p.is_file():
                out.append((cam_dir.name, p))
    return out


def save_overlay(path: Path, image_bgr: np.ndarray, points: np.ndarray, color=(0, 255, 0)):
    import cv2
    img = image_bgr.copy()
    for (x, y) in points.astype(int):
        cv2.circle(img, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def undistort_points(pts: np.ndarray, K: np.ndarray, dist: Optional[np.ndarray]) -> np.ndarray:
    """Undistort pixel points to the coordinate system of cv2.undistort(img, K, dist).

    Uses cv2.undistortPoints with P=K to return pixel coordinates aligned with the undistorted image.
    """
    import cv2
    if pts is None or pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    d = np.asarray(dist, dtype=np.float32).reshape(-1) if dist is not None else np.zeros(5, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    ud = cv2.undistortPoints(pts, K, d, P=K)
    return ud.reshape(-1, 2).astype(np.float32)
