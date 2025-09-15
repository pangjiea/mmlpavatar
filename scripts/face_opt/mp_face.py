from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Face2D:
    keypoints: np.ndarray  # (K, 2) pixel coordinates
    score: float
    bbox: Tuple[int, int, int, int]  # x,y,w,h


def detect_face_landmarks(image_bgr: np.ndarray, min_conf: float = 0.5) -> List[Face2D]:
    """Use MediaPipe Face Mesh to extract 468 2D landmarks.

    Returns a list of Face2D; usually at most one per image.
    """
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    img_h, img_w = image_bgr.shape[:2]
    rgb = image_bgr[:, :, ::-1]
    faces: List[Face2D] = []
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=min_conf) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return faces
        for lmks in res.multi_face_landmarks:
            pts = []
            xs, ys = [], []
            for lm in lmks.landmark:
                x = lm.x * img_w
                y = lm.y * img_h
                pts.append([x, y])
                xs.append(x)
                ys.append(y)
            pts = np.array(pts, dtype=np.float32)
            x0, x1 = int(np.clip(np.min(xs), 0, img_w-1)), int(np.clip(np.max(xs), 0, img_w-1))
            y0, y1 = int(np.clip(np.min(ys), 0, img_h-1)), int(np.clip(np.max(ys), 0, img_h-1))
            bbox = (x0, y0, x1-x0+1, y1-y0+1)
            faces.append(Face2D(keypoints=pts, score=1.0, bbox=bbox))
    return faces


def face_is_frontalish(face: Face2D, img_shape, min_rel_size: float = 0.05) -> bool:
    h, w = img_shape[:2]
    _, _, bw, bh = face.bbox
    area = bw * bh
    rel = area / float(w * h)
    return rel >= min_rel_size


def select_face_keypoints_subset(pts468: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Return a small, stable subset indices from the 468 mesh for robust PnP.

    We choose: eye corners, nose tip, mouth corners.
    Indices are based on MediaPipe's canonical mesh:
    - Right eye outer: 33
    - Left eye outer: 263
    - Nose tip: 1 (approx)
    - Mouth right corner: 61
    - Mouth left corner: 291
    """
    idx = [33, 263, 1, 61, 291]
    sub = pts468[idx]
    return sub, idx

