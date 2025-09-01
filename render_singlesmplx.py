import os
import json
import numpy as np
import torch
import imageio.v3 as iio
from argparse import ArgumentParser
from omegaconf import OmegaConf

from scene.net_vis import load_model
from utils.config_utils import Config
from utils.smpl_utils import init_smpl_pose, init_smpl


def fovx_to_intrinsic(fovx_rad: float, H: int, W: int) -> np.ndarray:
    focal = W / 2.0 / np.tan(fovx_rad / 2.0)
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = focal; K[1, 1] = focal; K[2, 2] = 1.0
    K[0, 2], K[1, 2] = W / 2.0, H / 2.0
    return K


@torch.no_grad()
def main():
    ap = ArgumentParser(description='Render single SMPL-X frame using saved cam + SMPL-X params')
    ap.add_argument('--config', type=str, required=True, help='YAML config path (same format as test.py)')
    ap.add_argument('--model_dir', type=str, required=True, help='Model directory with checkpoint')
    ap.add_argument('--cam_path', type=str, required=True, help='Saved cam JSON from viewer save')
    ap.add_argument('--smpl_path', type=str, required=True, help='Saved SMPL-X NPZ from viewer save')
    ap.add_argument('--out_path', type=str, required=True, help='Output image path (PNG)')
    args_cli = ap.parse_args()

    cfg: Config = OmegaConf.load(args_cli.config)

    # Initialize SMPL pose templates (required before loading model)
    if hasattr(cfg, 'smpl_pkl_path') and cfg.smpl_pkl_path:
        try:
            init_smpl(cfg.smpl_pkl_path)
        except Exception:
            init_smpl_pose()
    else:
        init_smpl_pose()

    # Load model
    gaussians = load_model(args_cli.model_dir)
    gaussians.is_test = True
    if hasattr(gaussians, 'prepare_test'):
        gaussians.prepare_test()

    # Background color (RGB in [0,1]) → torch CUDA
    bg = torch.as_tensor(np.array(cfg.background), dtype=torch.float32).cuda()

    # Load cam JSON (saved by viewer)
    with open(args_cli.cam_path, 'r') as f:
        cam_j = json.load(f)
    H, W = int(cam_j['height']), int(cam_j['width'])
    fovx_deg = float(cam_j['fovx'])
    w2c = torch.as_tensor(np.array(cam_j['w2c']).reshape(4, 4), dtype=torch.float32).cuda()
    K = torch.as_tensor(fovx_to_intrinsic(fovx_deg / 180.0 * np.pi, H, W), dtype=torch.float32).cuda()

    cam = {
        'w2c': w2c,
        'K': K,
        'height': H,
        'width': W,
    }

    # Load SMPL-X params (pose165, Th, Rh[, beta])
    smpl_npz = np.load(args_cli.smpl_path)
    pose = torch.as_tensor(smpl_npz['pose']).float()
    Th = torch.as_tensor(smpl_npz['Th']).float()
    Rh = torch.as_tensor(smpl_npz['Rh']).float()

    gaussians.smpl_poses = pose.cpu()
    gaussians.Th = Th.cpu()
    gaussians.Rh = Rh.cpu()

    image, alpha, info = gaussians.render(cam, background=bg)
    image = (torch.clamp(image, 0, 1.0) * 255).byte().contiguous().cpu().numpy()

    os.makedirs(os.path.dirname(args_cli.out_path) or '.', exist_ok=True)
    iio.imwrite(args_cli.out_path, image)
    print(f'[Render] Wrote {args_cli.out_path}')


if __name__ == '__main__':
    main()
