import os
import warnings
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
from utils.loss_utils import l1_loss as l1_loss_fn, psnr as psnr_fn, ssim_loss as ssim_loss_fn, lpips_loss as lpips_loss_fn
from utils.smpl_utils import init_smpl_pose

# Suppress FutureWarning from torchmetrics LPIPS weight loading
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torchmetrics\.functional\.image\.lpips")

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

def load_smpl_params_with_rhth(pose_path):
    """Load SMPL parameters with proper Rh and Th handling (consistent with dataset.py)"""
    smpl_params = np.load(pose_path, allow_pickle=True)
    smpl_params = dict(smpl_params)

    # 支持不同的数据集格式
    if 'Rh' in smpl_params:
        N_frame = len(smpl_params['Rh'])
    elif 'Th' in smpl_params:
        N_frame = len(smpl_params['Th'])
    elif 'global_orient' in smpl_params:
        N_frame = len(smpl_params['global_orient'])
    else:
        raise ValueError("无法确定帧数，SMPL参数文件中缺少Rh、Th或global_orient")

    pose_list = []
    for frame_id in range(N_frame):
        global_orient = smpl_params['global_orient'][frame_id] if 'global_orient' in smpl_params else np.zeros(3, dtype=np.float32)
        jaw_pose = smpl_params['jaw_pose'][frame_id] if 'jaw_pose' in smpl_params else np.zeros(3, dtype=np.float32)
        expression = smpl_params['expression'][frame_id] if 'expression' in smpl_params else np.zeros(10, dtype=np.float32)
        leye_pose = smpl_params['leye_pose'][frame_id] if 'leye_pose' in smpl_params else np.zeros(3, dtype=np.float32)
        reye_pose = smpl_params['reye_pose'][frame_id] if 'reye_pose' in smpl_params else np.zeros(3, dtype=np.float32)
        # 规范 expression 维度并保证 float32
        if len(expression) > 10:
            expression = expression[:10]
        elif len(expression) < 10:
            expression = np.pad(expression, (0, 10 - len(expression)))
        expression = np.asarray(expression, dtype=np.float32)

        pose = np.concatenate([
            global_orient.astype(np.float32),
            smpl_params['body_pose'][frame_id].astype(np.float32),
            jaw_pose.astype(np.float32),
            leye_pose.astype(np.float32),
            reye_pose.astype(np.float32),
            smpl_params['left_hand_pose'][frame_id].astype(np.float32),
            smpl_params['right_hand_pose'][frame_id].astype(np.float32),
        ], axis=0).astype(np.float32)

        if 'Th' in smpl_params:
            Th = smpl_params['Th'][frame_id].astype(np.float32)
        else:
            Th = smpl_params['transl'][frame_id].astype(np.float32)
        if 'Rh' in smpl_params:
            Rh = smpl_params['Rh'][frame_id].astype(np.float32)
        else:
            Rh = np.eye(3, dtype=np.float32)

        pose_list.append(dict(pose=pose, Th=Th, Rh=Rh, expression=expression.astype(np.float32)))
    return pose_list
#原先加载npz方法，但缺少expression jawpose eyepose 目前不调用
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
    gaussians.smpl_poses = torch.as_tensor(pose['pose'], dtype=torch.float32)
    gaussians.Th = torch.as_tensor(pose['Th'], dtype=torch.float32)
    gaussians.Rh = torch.as_tensor(pose['Rh'], dtype=torch.float32)
    #  获取不到 expression 时打印
    if 'expression' in pose and pose['expression'] is not None:
        expr = pose['expression']
    else:
        print('[Info] expression 缺失, 使用零向量 (warmup frame 0)')
        expr = torch.zeros(10, dtype=torch.float32)
    gaussians.expression = torch.as_tensor(expr, dtype=torch.float32)
    gaussians.jaw_pose = gaussians.smpl_poses[66:69]
    gaussians.leye_pose = gaussians.smpl_poses[69:72]
    gaussians.reye_pose = gaussians.smpl_poses[72:75]
    image, alpha, info = gaussians.render(cam, background=background)
    torch.cuda.synchronize()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    iter_start.record()

    for frame_id in frame_ids:
        pose = pose_list[frame_id]
        gaussians.smpl_poses = torch.as_tensor(pose['pose'], dtype=torch.float32)
        gaussians.Th = torch.as_tensor(pose['Th'], dtype=torch.float32)
        gaussians.Rh = torch.as_tensor(pose['Rh'], dtype=torch.float32)
        # 表情与眼睛控制（若不存在则使用默认0并打印）
        if 'expression' in pose and pose['expression'] is not None:
            expr = pose['expression']
        else:
            print(f'[Info] expression 缺失, 使用零向量 (frame {frame_id})')
            expr = torch.zeros(10, dtype=torch.float32)
        gaussians.expression = torch.as_tensor(expr, dtype=torch.float32)
        gaussians.jaw_pose = gaussians.smpl_poses[66:69]
        gaussians.leye_pose = gaussians.smpl_poses[69:72]
        gaussians.reye_pose = gaussians.smpl_poses[72:75]

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

        gaussians.smpl_poses = torch.as_tensor(pose['pose'], dtype=torch.float32)
        gaussians.Th = torch.as_tensor(pose['Th'], dtype=torch.float32)
        gaussians.Rh = torch.as_tensor(pose['Rh'], dtype=torch.float32)
        # 修改: 获取不到 expression 时打印
        if 'expression' in pose and pose['expression'] is not None:
            expr = pose['expression']
        else:
            print(f'[Info] expression 缺失, 使用零向量 (frame {frame_id})')
            expr = torch.zeros(10, dtype=torch.float32)
        gaussians.expression = torch.as_tensor(expr, dtype=torch.float32)
        gaussians.jaw_pose = gaussians.smpl_poses[66:69]
        gaussians.leye_pose = gaussians.smpl_poses[69:72]
        gaussians.reye_pose = gaussians.smpl_poses[72:75]
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
    l1_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    num = 0

    # Optional FID (skip gracefully if torchmetrics/weights unavailable)
    fid = None
    try:
        from torchmetrics.image import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048).cuda()
    except Exception as e:
        print(f"[info] FID unavailable, skipping: {e}")

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        gaussians.smpl_poses = cam['pose'].float()
        gaussians.Th, gaussians.Rh = cam['Th'].float(), cam['Rh'].float()
        # 表情与眼睛控制 (缺失则打印)
        if 'expression' in cam and cam['expression'] is not None:
            expr = cam['expression'].float()
        else:
            print(f'[Info] expression 缺失, 使用零向量 (dataset frame {frame_id})')
            expr = torch.zeros(10, dtype=torch.float32)
        gaussians.expression = torch.as_tensor(expr, dtype=torch.float32)
        gaussians.jaw_pose = gaussians.smpl_poses[66:69]
        gaussians.leye_pose = gaussians.smpl_poses[69:72]
        gaussians.reye_pose = gaussians.smpl_poses[72:75]

        image, alpha, info = gaussians.render(cam, background=background)
        image = torch.clamp(image, min=0.0, max=1.0)

        image_gt = cam['image']
        image_gt[~cam['mask']] = background

        # Metrics on float tensors
        try:
            l1_sum += l1_loss_fn(image, image_gt).mean().float()
            psnr_sum += psnr_fn(image, image_gt).mean().float()
            ssim_sum += (1.0 - ssim_loss_fn(image, image_gt)).mean().float()
            lpips_sum += lpips_loss_fn(image, image_gt).mean().float()
            num += 1
        except Exception as e:
            if num == 0:
                print(f"[info] image metrics unavailable, skipping some: {e}")

        # FID expects uint8 CHW batches
        if fid is not None:
            try:
                pred_chw = (image.permute(2,0,1)[None] * 255.0).clamp(0,255).byte()
                gt_chw = (image_gt.permute(2,0,1)[None] * 255.0).clamp(0,255).byte()
                fid.update(pred_chw, real=False)
                fid.update(gt_chw, real=True)
            except Exception as e:
                print(f"[info] FID update failed for frame {int(frame_id)}: {e}")

        # Convert to numpy for saving
        image_np = (image * 255).byte().contiguous().cpu().numpy()
        image_gt_np = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt_np)
        iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image_np)
        iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)

    # Summarize metrics
    if num > 0:
        l1_mean = l1_sum / num
        psnr_mean = psnr_sum / num
        ssim_mean = ssim_sum / num
        lpips_mean = lpips_sum / num
    else:
        l1_mean = psnr_mean = ssim_mean = lpips_mean = None

    fid_val = None
    if fid is not None:
        try:
            fid_val = fid.compute()
        except Exception as e:
            print(f"[info] FID compute failed: {e}")

    # Print summary
    msg = "[TEST] Metrics:"
    if l1_mean is not None:
        msg += f" L1 {l1_mean}"
    if psnr_mean is not None:
        msg += f" PSNR {psnr_mean}"
    if ssim_mean is not None:
        msg += f" SSIM {ssim_mean}"
    if lpips_mean is not None:
        msg += f" LPIPS {lpips_mean}"
    if fid_val is not None:
        msg += f" FID {fid_val}"
    print(msg)


@torch.no_grad()
def testing(args: Config):
    init_smpl_pose()

    gaussians = load_model(args.model_dir)
    # 强制 encoder_feat_params 转为 float32 避免 Double/Float 混用
    if hasattr(gaussians, 'encoder_feat_params') and isinstance(gaussians.encoder_feat_params, dict):
        for k in list(gaussians.encoder_feat_params.keys()):
            gaussians.encoder_feat_params[k] = gaussians.encoder_feat_params[k].float()
    gaussians.is_test = args.test.is_test
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(args.background), dtype=torch.float32).cuda()

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
            pose_list = load_smpl_params_with_rhth(args.test.pose_path)
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
