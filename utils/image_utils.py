
import torch
import numpy as np
import cv2 as cv
import math
import warnings

import torch
import torch.nn.functional as F

from torchvision.io import encode_jpeg
from utils.smpl_utils import smpl
import torch
import numpy as np

def calc_bbox(mask, margin=0):
    # [left right)  [top down)
    height, width = mask.shape
    mask_sum0 = np.sum(mask, axis=0) > 0
    mask_sum1 = np.sum(mask, axis=1) > 0
    left = np.argmax(mask_sum0)
    right = width - np.argmax(mask_sum0[::-1])
    top = np.argmax(mask_sum1)
    down = height - np.argmax(mask_sum1[::-1])

    if margin != 0:
        left = max(0, left - margin)
        right = min(width, right + margin)
        top = max(0, top - margin)
        down = min(height, down + margin)
    bbox = np.array([left, top, right, down], dtype=int)
    return bbox

try:
    from nvjpeg import NvJpeg
    nj = NvJpeg()
except:
    pass 

def encode_bytes(image, image_encode_method=''):
    if image_encode_method == 'cpu':
        image_byte = cv.imencode('.jpg', image)
    elif image_encode_method == 'gpu':
        image_byte = nj.encode(image, 90)
    elif image_encode_method == 'torch':

        if torch.__version__ < "2.4.0":
            warnings.warn(
                "Warning: torch version is less than 2.4.0. "
                "GPU encoding of JPEG images using encode_jpeg is only available in torch version 2.4.0 or higher.",
                UserWarning
            )

        image = torch.flip(image, dims=[2]).permute(2,0,1)
        torch.cuda.synchronize()
        image_byte = encode_jpeg(image, 90).cpu().numpy().tobytes()
    else:
        image_byte = image
    return image_byte

# code from AnimatableGaussians https://github.com/lizhe00/AnimatableGaussians/blob/master/main_avatar.py
def crop_image(bg_color_cuda, gt_mask, patch_size, randomly, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param randomly: whether to randomly sample the patch
    :param args: input images with shape of (C, H, W)
    """
    mask_uv = torch.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)[0]
    max_v, max_u = mask_uv.max(0)[0]
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    if randomly and max_size > patch_size:
        random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
    for image in args:
        cropped_image = bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
        if len_v > len_u:
            start_u = (max_size - len_u) // 2
            cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
        else:
            start_v = (max_size - len_v) // 2
            cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

        if randomly and max_size > patch_size:
            cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
        else:
            cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
        cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]


def calc_face_bbox(mask: torch.Tensor,
                   top_frac: float = 0.15,
                   height_frac: float = 0.22,
                   width_frac: float = 0.35,
                   min_size: int = 32):
    """Compute a coarse face bbox from a person mask.

    Args:
        mask: HxW bool tensor (CPU or CUDA).
        top_frac: start position inside person bbox from top (0-1).
        height_frac: fraction of person bbox height to use for face bbox.
        width_frac: fraction of person bbox width to use for face bbox.
        min_size: minimum bbox size in pixels.

    Returns:
        bbox = (left, top, right, bottom) ints, or None if mask empty.
    """
    if mask.numel() == 0:
        return None
    # Work on CPU for indexing convenience
    if mask.is_cuda:
        m = mask.detach().to('cpu')
    else:
        m = mask.detach()
    if m.dtype != torch.bool:
        m = m > 0
    ys, xs = torch.nonzero(m, as_tuple=True)
    if ys.numel() == 0:
        return None
    top = int(ys.min().item()); bottom = int(ys.max().item()) + 1
    left = int(xs.min().item()); right = int(xs.max().item()) + 1
    h = max(1, bottom - top); w = max(1, right - left)

    # Face ROI parameters
    face_h = max(min_size, int(round(h * float(height_frac))))
    face_w = max(min_size, int(round(w * float(width_frac))))
    face_top = top + int(round(h * float(top_frac)))
    face_left = left + (w - face_w) // 2

    # Clamp to image bounds
    H, W = m.shape
    face_top = max(0, min(H - face_h, face_top))
    face_left = max(0, min(W - face_w, face_left))
    face_bottom = min(H, face_top + face_h)
    face_right = min(W, face_left + face_w)

    if face_bottom <= face_top or face_right <= face_left:
        return None
    return (face_left, face_top, face_right, face_bottom)


def mask_from_bbox(shape_hw, bbox):
    """Create a boolean mask HxW from bbox=(l,t,r,b)."""
    if bbox is None:
        return None
    H, W = shape_hw
    l, t, r, b = bbox
    m = torch.zeros((H, W), dtype=torch.bool)
    m[t:b, l:r] = True
    return m


def _project_points(K: np.ndarray, w2c: np.ndarray, pts_world: np.ndarray):
    """Project Nx3 world points to pixel coordinates using intrinsics and w2c."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ pts_world.T + t[:, None]).T  # Nx3
    z = pts_cam[:, 2:3]
    z = np.clip(z, 1e-6, None)
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / z
    return uv, pts_cam[:, 2]


def calc_face_bbox_smplx(pose_vec: torch.Tensor,
                         beta_vec: torch.Tensor,
                         expression_vec: torch.Tensor,
                         jaw_pose_vec: torch.Tensor,
                         Rh: torch.Tensor,
                         Th: torch.Tensor,
                         K: torch.Tensor,
                         w2c: torch.Tensor,
                         image_hw: tuple,
                         radius_scale: float = 2.2,
                         min_size: int = 32):
    """Estimate a face bbox by projecting a 3D head-centered rectangle derived from SMPL-X joints.

    Returns bbox=(l,t,r,b) or None.
    """
    try:
        # Prepare SMPL-X inputs (SQ02: global_orient=0, transl=0)
        body_pose = pose_vec[3:66].detach().cpu().unsqueeze(0)  # [1,63]
        betas = beta_vec.detach().cpu().unsqueeze(0)  # [1,10]
        jaw_pose = jaw_pose_vec.detach().cpu().unsqueeze(0)  # [1,3]
        # Expression may be >10, clamp to 10
        expr = expression_vec.detach().cpu()
        if expr.numel() > 10:
            expr = expr[:10]
        elif expr.numel() < 10:
            expr = torch.nn.functional.pad(expr, (0, 10 - expr.numel()))
        expr = expr.unsqueeze(0)

        smpl_out = smpl.model(
            betas=betas,
            body_pose=body_pose,
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
            expression=expr,
            jaw_pose=jaw_pose,
        )
        joints = smpl_out.joints[0].detach().cpu().numpy()

        # Joint indices (SMPL layout used across repo)
        neck_idx = 12
        head_idx = 15
        if max(neck_idx, head_idx) >= joints.shape[0]:
            return None
        neck = joints[neck_idx]
        head = joints[head_idx]
        center = head
        base = np.linalg.norm(head - neck) + 1e-6
        r = max(min_size * 0.5, radius_scale * base)

        # Apply global transform: x' = R @ x + T
        Rg = Rh.detach().cpu().numpy()
        Tg = Th.detach().cpu().numpy()
        center_w = (Rg @ center.reshape(3, 1)).reshape(3) + Tg

        # Camera basis in world
        w2c_np = w2c.detach().cpu().numpy()
        K_np = K.detach().cpu().numpy()
        R_wc = w2c_np[:3, :3]
        R_cw = R_wc.T
        right = R_cw[:, 0]
        up = R_cw[:, 1]

        # Corners around head center in world space
        offsets = [
            +r * right + r * up,
            +r * right - r * up,
            -r * right + r * up,
            -r * right - r * up,
        ]
        pts = np.stack([center_w + o for o in offsets], axis=0)
        uv, depth = _project_points(K_np, w2c_np, pts)

        # Discard if all behind camera
        if np.all(depth <= 0):
            return None
        umin = int(np.floor(np.min(uv[:, 0])))
        umax = int(np.ceil(np.max(uv[:, 0])))
        vmin = int(np.floor(np.min(uv[:, 1])))
        vmax = int(np.ceil(np.max(uv[:, 1])))

        H, W = image_hw
        l = max(0, min(W, umin))
        r_ = max(0, min(W, umax))
        t = max(0, min(H, vmin))
        b = max(0, min(H, vmax))
        if r_ - l < min_size or b - t < min_size:
            # Expand to min size around center projection
            center_uv, _ = _project_points(K_np, w2c_np, center_w[None, :])
            cu, cv = center_uv[0]
            l = int(max(0, min(W, cu - min_size / 2)))
            r_ = int(max(0, min(W, cu + min_size / 2)))
            t = int(max(0, min(H, cv - min_size / 2)))
            b = int(max(0, min(H, cv + min_size / 2)))

        if r_ <= l or b <= t:
            return None
        return (l, t, r_, b)
    except Exception:
        return None


def calc_face_mask_smplx(pose_vec: torch.Tensor,
                         beta_vec: torch.Tensor,
                         expression_vec: torch.Tensor,
                         jaw_pose_vec: torch.Tensor,
                         Rh: torch.Tensor,
                         Th: torch.Tensor,
                         K: torch.Tensor,
                         w2c: torch.Tensor,
                         image_hw: tuple,
                         radius_scale: float = 2.2,
                         ax_min_frac: float = 0.25,
                         ax_max_frac: float = 2.0,
                         min_size: int = 32):
    """Project SMPL-X head region to image and return a binary mask (uint8 0/255) and bbox.

    The head region is approximated by vertices above the neck along the head axis and
    within a cylinder radius proportional to neck-head distance.
    Returns (mask_np_uint8, bbox) or (None, None) on failure.
    """
    try:
        # Prepare SMPL-X inputs (SQ02: global_orient=0, transl=0)
        body_pose = pose_vec[3:66].detach().cpu().unsqueeze(0)  # [1,63]
        betas = beta_vec.detach().cpu().unsqueeze(0)  # [1,10]
        jaw_pose = jaw_pose_vec.detach().cpu().unsqueeze(0)  # [1,3]
        expr = expression_vec.detach().cpu()
        if expr.numel() > 10:
            expr = expr[:10]
        elif expr.numel() < 10:
            expr = torch.nn.functional.pad(expr, (0, 10 - expr.numel()))
        expr = expr.unsqueeze(0)

        smpl_out = smpl.model(
            betas=betas,
            body_pose=body_pose,
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
            expression=expr,
            jaw_pose=jaw_pose,
        )
        verts = smpl_out.vertices[0].detach().cpu().numpy()  # (V,3)
        joints = smpl_out.joints[0].detach().cpu().numpy()

        # Neck-head axis
        neck_idx = 12
        head_idx = 15
        if max(neck_idx, head_idx) >= joints.shape[0]:
            return None, None
        neck = joints[neck_idx]
        head = joints[head_idx]
        axis = head - neck
        base = np.linalg.norm(axis) + 1e-6
        u = axis / base

        # Select head vertices by axial distance and radial distance to axis
        v_rel = verts - neck[None, :]
        d_ax = v_rel @ u  # projection length along axis
        v_par = np.outer(d_ax, u)
        v_perp = v_rel - v_par
        r_perp = np.linalg.norm(v_perp, axis=1)

        min_ax = float(ax_min_frac) * base         # above neck
        max_ax = float(ax_max_frac) * base         # not too far beyond head top
        radius = float(radius_scale) * base
        sel = (d_ax >= min_ax) & (d_ax <= max_ax) & (r_perp <= radius)

        if not np.any(sel):
            return None, None

        # Apply global Rh/Th
        Rg = Rh.detach().cpu().numpy()
        Tg = Th.detach().cpu().numpy()
        verts_sel = verts[sel]
        verts_w = (Rg @ verts_sel.T + Tg[:, None]).T

        # Project to pixels
        w2c_np = w2c.detach().cpu().numpy()
        K_np = K.detach().cpu().numpy()
        uv, depth = _project_points(K_np, w2c_np, verts_w)
        valid = depth > 0
        uv = uv[valid]
        if uv.shape[0] < 8:
            return None, None

        H, W = image_hw
        pts = uv.astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        # Convex hull and fill
        hull = cv.convexHull(pts.reshape(-1, 1, 2).astype(np.float32))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.fillConvexPoly(mask, hull.astype(np.int32), 255)

        # BBox
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None, None
        t, b = int(ys.min()), int(ys.max()) + 1
        l, r = int(xs.min()), int(xs.max()) + 1
        if (b - t) < min_size or (r - l) < min_size:
            # expand around center if too small
            cy = (t + b) // 2
            cx = (l + r) // 2
            half = min_size // 2
            t = max(0, cy - half)
            b = min(H, cy + half)
            l = max(0, cx - half)
            r = min(W, cx + half)
        return mask, (l, t, r, b)
    except Exception:
        return None, None
