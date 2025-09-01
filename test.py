
import os
from os import path
import torch
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import numpy as np
import random
import pickle
import copy
import json
from argparse import ArgumentParser
import copy
from scipy.spatial.transform import Rotation
import imageio.v3 as iio
from torch.utils.data import DataLoader

from scene.dataset import get_dataset_type, data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import load_model
from utils.config_utils import Config
from utils.image_utils import encode_bytes
from utils.smpl_utils import init_smpl_pose, init_smpl, smpl
from pytorch3d.ops import knn_points
import math

def fovx_to_intrinsic(fovx, H, W):
    focal = W / 2 / np.tan(fovx/2)
    K = np.zeros((3, 3))
    K[0, 0] = focal
    K[1, 1] = focal
    K[2, 2] = 1
    K[0, 2], K[1, 2] = W/2, H/2
    return K.astype(np.float32)

def load_amass_pose_list(pose_path):
    data = np.load(pose_path)
    pose_list = []
    poses = data['poses'].astype(np.float32)
    trans = data['trans'].astype(np.float32)
    N = len(poses)

    # AMASS poses are noisy
    OPTIMIZE_AMASS = False
    if OPTIMIZE_AMASS:
        foo = poses[:,3:]
        foo[:, 13 * 3 + 2] -= 0.25
        foo[:, 12 * 3 + 2] += 0.25
        foo[:, 19 * 3: 20 * 3] = 0.
        foo[:, 20 * 3: 21 * 3] = 0.
        foo[:, 14 * 3] = 0.

        poses[:,3:] = foo

        # smooth
        win_size = 1
        poses_clone = np.copy(poses)
        trans_clone = np.copy(trans)
        frame_num = poses_clone.shape[0]
        poses[win_size: frame_num-win_size] = 0
        trans[win_size: frame_num-win_size] = 0
        for i in range(-win_size, win_size + 1):
            poses[win_size: frame_num-win_size] += poses_clone[win_size+i: frame_num-win_size+i]
            trans[win_size: frame_num-win_size] += trans_clone[win_size+i: frame_num-win_size+i]
        poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
        trans[win_size: frame_num-win_size] /= (2 * win_size + 1)

    for i in range(N):
        pose_list.append(dict(pose=poses[i], Th=trans[i], Rh=np.eye(3, dtype=np.float32)))

    return pose_list

def load_thuman_pose_list(pose_path):
    smpl_params = np.load(pose_path, allow_pickle=True)
    smpl_params = dict(smpl_params)

    pose_list = []
    N = len(smpl_params['global_orient'])
    for frame_id in range(N):
        pose = np.concatenate([smpl_params['global_orient'][frame_id],
                    smpl_params['body_pose'][frame_id],
                    np.zeros(3,dtype=np.float32),
                    np.zeros(6,dtype=np.float32),
                    smpl_params['left_hand_pose'][frame_id],
                    smpl_params['right_hand_pose'][frame_id],], axis=0)
        Th = smpl_params['transl'][frame_id]
        Rh = np.eye(3, dtype=np.float32)
        pose_list.append(dict(pose=pose, Th=Th, Rh=Rh))
    return pose_list

def testing_novel_cam_pose_speed(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background):

    # warm up
    pose = pose_list[0]
    gaussians.smpl_poses = torch.as_tensor(pose['pose'])
    gaussians.Th = torch.as_tensor(pose['Th'])
    gaussians.Rh = torch.as_tensor(pose['Rh'])
    image, alpha, info = gaussians.render(cam, background=background)
    torch.cuda.synchronize()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    iter_start.record()

    for frame_id in frame_ids:
        pose = pose_list[frame_id]
        gaussians.smpl_poses = torch.as_tensor(pose['pose'])
        gaussians.Th = torch.as_tensor(pose['Th'])
        gaussians.Rh = torch.as_tensor(pose['Rh'])

        image, alpha, info = gaussians.render(cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous()
        torch.cuda.synchronize()

    iter_end.record()
    torch.cuda.synchronize()

    run_time = iter_start.elapsed_time(iter_end)
    fps = len(frame_ids) / run_time * 1000
    print('Running time:', run_time)
    print('FPS:', fps)

def testing_novel_cam_pose(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background):

    os.makedirs(path.join(out_dir), exist_ok=True)
    for frame_id in tqdm(frame_ids):
        pose = pose_list[frame_id]
        pose = copy.deepcopy(pose)

        gaussians.smpl_poses = torch.as_tensor(pose['pose']).cpu()
        gaussians.Th = torch.clone(torch.as_tensor(pose['Th']).cpu())
        gaussians.Rh = torch.as_tensor(pose['Rh']).cpu()
        image, alpha, info = gaussians.render(cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
        iio.imwrite(path.join(out_dir, f'{frame_id:08d}.png'), image)


def testing_dataset(gaussians: GaussianModel, out_dir, dataset, background):
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    for k in ['gt', 'result', 'mask']:
        os.makedirs(path.join(out_dir, k), exist_ok=True)

    # Metrics accumulators
    iou_list = []
    # Optional: boundary F-score lists can be added later
    chamfer_l2_cm2_list = []   # Chamfer-L2 in cm^2
    chamfer_l1_cm_list = []    # Chamfer-L1 in cm
    thr_recall_pred2gt_list = []  # fraction of pred within 0.8cm to GT
    thr_recall_gt2pred_list = []  # fraction of GT within 0.8cm to pred
    processed_frames_for_3d = set()

    # Simple cache to avoid recomputing GT vertices per frame
    gt_vertices_cache = {}

    def compute_iou_from_alpha(alpha_t: torch.Tensor, gt_mask_t: torch.Tensor, thr: float = 0.5) -> float:
        """Robust IoU between predicted alpha and GT mask.
        - Accept alpha as HxW, HxW×1, or HxWxC. If multi-channel, take first channel.
        - Accept gt_mask as bool or uint8 tensor of shape HxW or HxW×1.
        """
        # Ensure 2D alpha
        a = alpha_t
        if a.ndim == 3:
            # HxWxC → take first channel
            a = a[..., 0]
        elif a.ndim > 3:
            a = a.squeeze()
            if a.ndim == 3:
                a = a[..., 0]
        # Ensure 2D mask
        m = gt_mask_t
        if m.ndim == 3:
            m = m[..., 0]
        elif m.ndim > 3:
            m = m.squeeze()
            if m.ndim == 3:
                m = m[..., 0]
        # Types
        pred = (a > thr)
        gt = m.bool()
        # Guard resize if shapes mismatch (should not happen if dataset/render align)
        if pred.shape != gt.shape:
            # Fallback: center-crop to min shape
            H = min(pred.shape[0], gt.shape[0])
            W = min(pred.shape[1], gt.shape[1])
            pred = pred[:H, :W]
            gt = gt[:H, :W]
        inter = (pred & gt).sum().item()
        union = (pred | gt).sum().item()
        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return inter / union

    def get_smpl_vertices_world(pose_165: torch.Tensor, beta_10: torch.Tensor, Rh: torch.Tensor, Th: torch.Tensor) -> torch.Tensor:
        """Compute SMPL-X vertices in world coordinates matching GaussianModel's Rh/Th application.
        - pose_165: (165,) axis-angle SMPL-X order used in this repo
        - beta_10: (10,)
        - Rh: (3,3), Th: (3,)
        Returns: (V,3) torch.float32 on CUDA
        """
        device = torch.device('cpu')  # build on CPU then move
        p = pose_165.detach().cpu().float()
        b = beta_10.detach().cpu().float()
        # Split poses
        go = p[0:3][None]                 # (1,3)
        body = p[3: 3+21*3][None]         # (1,63)
        jaw = p[3+21*3: 3+21*3+3][None]   # (1,3)
        # skip eyes (zeros)
        lh = p[3+21*3+3+6 : 3+21*3+3+6 + 15*3][None]   # (1,45)
        rh = p[3+21*3+3+6 + 15*3 : 3+21*3+3+6 + 30*3][None]  # (1,45)

        out = smpl.model(
            betas=b[None],
            global_orient=go,
            body_pose=body,
            jaw_pose=jaw,
            left_hand_pose=lh,
            right_hand_pose=rh,
            transl=None,
            return_verts=True,
        )
        verts = out.vertices[0].detach().cpu()  # (V,3)
        # Apply Rh/Th to align with GaussianModel
        Rh_cpu = Rh.detach().cpu().float()
        Th_cpu = Th.detach().cpu().float()
        verts_w = (Rh_cpu @ verts.T).T + Th_cpu
        return verts_w.cuda(non_blocking=True)

    def compute_chamfer_cm(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor, thr_cm: float = 0.8):
        """Compute Chamfer metrics between two point sets on CUDA.
        Returns (chamfer_l2_cm2, chamfer_l1_cm, recall_pred2gt, recall_gt2pred)
        """
        # pred_xyz, gt_xyz: (N,3), (M,3) on CUDA
        with torch.no_grad():
            d2_p2g = knn_points(pred_xyz[None], gt_xyz[None], K=1)[0][0,:,0]  # meters^2
            d2_g2p = knn_points(gt_xyz[None], pred_xyz[None], K=1)[0][0,:,0]
            # convert to cm
            d_cm_p2g = torch.sqrt(d2_p2g) * 100.0
            d_cm_g2p = torch.sqrt(d2_g2p) * 100.0
            # L2 (squared in cm)
            chamfer_l2_cm2 = 0.5 * (torch.mean(d_cm_p2g**2) + torch.mean(d_cm_g2p**2)).item()
            # L1 (cm)
            chamfer_l1_cm = 0.5 * (torch.mean(d_cm_p2g) + torch.mean(d_cm_g2p)).item()
            thr = torch.tensor(thr_cm, device=pred_xyz.device, dtype=d_cm_p2g.dtype)
            recall_pred2gt = torch.mean((d_cm_p2g <= thr).float()).item()
            recall_gt2pred = torch.mean((d_cm_g2p <= thr).float()).item()
        return chamfer_l2_cm2, chamfer_l1_cm, recall_pred2gt, recall_gt2pred

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

        image, alpha, info = gaussians.render(cam, background=background)

        # Metrics: IoU per view
        try:
            iou = compute_iou_from_alpha(alpha, cam['mask'])
            iou_list.append(iou)
        except Exception:
            pass

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image']
        image_gt[~cam['mask']] = background
        image_gt = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt)
        iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image)
        iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)

        # Metrics: Chamfer on unique frames only (avoid repeating per-cam)
        if frame_id not in processed_frames_for_3d:
            try:
                # Prepare GT vertices (cache per frame)
                if frame_id not in gt_vertices_cache:
                    gt_vertices_cache[frame_id] = get_smpl_vertices_world(cam['pose'], cam['beta'], cam['Rh'], cam['Th'])
                gt_verts_cuda = gt_vertices_cache[frame_id]
                # Predicted GS positions for current pose
                pred_xyz = gaussians.get_xyz  # (N,3) CUDA
                l2_cm2, l1_cm, r_p2g, r_g2p = compute_chamfer_cm(pred_xyz, gt_verts_cuda, thr_cm=0.8)
                chamfer_l2_cm2_list.append(l2_cm2)
                chamfer_l1_cm_list.append(l1_cm)
                thr_recall_pred2gt_list.append(r_p2g)
                thr_recall_gt2pred_list.append(r_g2p)
                processed_frames_for_3d.add(frame_id)
            except Exception as e:
                # Skip if SMPL model not available or any failure
                pass

    # Summarize metrics
    summary = {}
    if iou_list:
        arr = np.array(iou_list, dtype=np.float32)
        summary['iou_mean'] = float(arr.mean())
        summary['iou_median'] = float(np.median(arr))
        summary['iou_p5'] = float(np.percentile(arr, 5))
        summary['iou_p95'] = float(np.percentile(arr, 95))
    if chamfer_l2_cm2_list:
        c2 = np.array(chamfer_l2_cm2_list, dtype=np.float32)
        c1 = np.array(chamfer_l1_cm_list, dtype=np.float32)
        r1 = np.array(thr_recall_pred2gt_list, dtype=np.float32)
        r2 = np.array(thr_recall_gt2pred_list, dtype=np.float32)
        summary['chamfer_L2_cm2_mean'] = float(c2.mean())
        summary['chamfer_L1_cm_mean'] = float(c1.mean())
        summary['recall_pred2gt_0.8cm'] = float(r1.mean())
        summary['recall_gt2pred_0.8cm'] = float(r2.mean())

    if summary:
        os.makedirs(out_dir, exist_ok=True)
        with open(path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('Test summary:', summary)


@torch.no_grad()
def testing(args: Config):
    # Ensure SMPL-X model is loaded for GT vertex generation
    if hasattr(args, 'smpl_pkl_path') and args.smpl_pkl_path:
        try:
            init_smpl(args.smpl_pkl_path)
        except Exception as e:
            print('Warning: init_smpl failed, 3D metrics will be skipped.', e)
    else:
        init_smpl_pose()

    gaussians = load_model(args.model_dir)
    gaussians.is_test = args.test.is_test
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(args.background)).float().cuda()

    # Dataset
    test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.frame_interval*args.test.num_frame, args.test.frame_interval).tolist()
    test_cam_ids = np.array(args.test.cam_ids).tolist()

    if args.test.cam_path is not None and args.test.pose_path is not None:
        with open(args.test.cam_path, 'r') as file:
            cam = json.load(file)
        cam['w2c'] = torch.as_tensor(np.array(cam['w2c']).reshape(4,4)).float().cuda()
        K = fovx_to_intrinsic(cam['fovx'] / 180 * np.pi, cam['height'], cam['width'])
        cam['K'] = torch.as_tensor(K).cuda()

        if 'smpl_params.npz' in args.test.pose_path:
            pose_list = load_thuman_pose_list(args.test.pose_path)
        else:
            pose_list = load_amass_pose_list(args.test.pose_path)

        if args.test.test_speed:
            testing_novel_cam_pose_speed(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background)
        else:
            testing_novel_cam_pose(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background)
    else:
        DatasetType = get_dataset_type(args.data_dir)
        testset = DatasetType(
            datadir=args.data_dir,
            frame_ids=test_frame_ids,
            cam_ids=test_cam_ids,
            background=np.array(args.background),
            image_scaling=args.image_scaling,
        )

        testing_dataset(gaussians, args.out_dir, testset, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')

    parser.add_argument('--cam_path', type=str, default=None)
    parser.add_argument('--pose_path', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_speed', action='store_true')
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir, args.model_dir, args.test.cam_path, args.test.pose_path = pargs.data_dir, pargs.out_dir, pargs.model_dir, pargs.cam_path, pargs.pose_path
    args.test.is_test, args.test.test_speed = pargs.test, pargs.test_speed
    torch.backends.cuda.matmul.allow_tf32 = True

    testing(args)
