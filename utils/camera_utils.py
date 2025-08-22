import math
import torch

def k_to_fovx_fovy(K: torch.Tensor, width: int, height: int):
    """Compute FoVx/FoVy (radians) from intrinsics and resolution.
    K: [3,3] with fx, fy, cx, cy in pixels.
    """
    fx = float(K[0, 0].item())
    fy = float(K[1, 1].item())
    fovx = 2.0 * math.atan2(width * 0.5, fx)
    fovy = 2.0 * math.atan2(height * 0.5, fy)
    return fovx, fovy

def invert_se3(w2c: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix (CUDA/Tensor friendly)."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    Rt = R.transpose(0, 1)
    inv = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
    inv[:3, :3] = Rt
    inv[:3, 3] = -Rt @ t
    return inv

def build_proj_from_k(K: torch.Tensor, width: int, height: int, znear=0.01, zfar=100.0) -> torch.Tensor:
    """Build a GL-like projection matrix consistent with intrinsics.
    This follows a standard pinhole-to-GL projection; TRasterizer expects a 4x4 proj.
    """
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]
    proj = torch.zeros((4, 4), device=K.device, dtype=K.dtype)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 1.0 - 2.0 * cx / width
    proj[1, 2] = 2.0 * cy / height - 1.0
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    proj[3, 2] = -1.0
    return proj

