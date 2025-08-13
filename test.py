
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
from utils.smpl_utils import init_smpl_pose

def fovx_to_intrinsic(fovx, H, W):
    focal = W / 2 / np.tan(fovx/2)
    K = np.zeros((3, 3))
    K[0, 0] = focal
    K[1, 1] = focal
    K[2, 2] = 1
    K[0, 2], K[1, 2] = W/2, H/2
    return K.astype(np.float32)

def load_json_pose_list(pose_path):
    """Load pose list from JSON file"""
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    
    pose_list = []
    for frame_data in pose_data:
        pose_list.append({
            'pose': np.array(frame_data['pose'], dtype=np.float32),
            'Th': np.array(frame_data['Th'], dtype=np.float32),
            'Rh': np.array(frame_data['Rh'], dtype=np.float32)
        })
    
    return pose_list

def load_amass_pose_list(pose_path):
    data = np.load(pose_path, allow_pickle=True)
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

def testing_360frame_novel_view(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam_trajectory, background):
    """Test with 360-frame novel view generation using camera trajectory"""
    
    os.makedirs(path.join(out_dir), exist_ok=True)
    
    print(f"Generating {len(frame_ids)} frames with {len(cam_trajectory)} camera poses...")
    
    for frame_id in tqdm(frame_ids):
        # Get pose for this frame
        pose = pose_list[frame_id % len(pose_list)]  # Cycle through poses if needed
        pose = copy.deepcopy(pose)
        
        # Get camera for this frame (cycle through camera trajectory)
        cam_idx = frame_id % len(cam_trajectory)
        cam = cam_trajectory[cam_idx]
        
        # Set SMPL parameters
        gaussians.smpl_poses = torch.as_tensor(pose['pose']).cpu()
        gaussians.Th = torch.clone(torch.as_tensor(pose['Th']).cpu())
        gaussians.Rh = torch.as_tensor(pose['Rh']).cpu()
        
        # Render image
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

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

        image, alpha, info = gaussians.render(cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image']
        image_gt[~cam['mask']] = background
        image_gt = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt)
        iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image)
        iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)


@torch.no_grad()
def testing(args: Config):
    init_smpl_pose()

    gaussians = load_model(args.model_dir)
    gaussians.is_test = args.test.is_test
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(args.background)).float().cuda()

    # Dataset
    if pargs.num_frames != 360:
        # Use custom number of frames for 360-view generation
        test_frame_ids = np.arange(pargs.num_frames).tolist()
    else:
        test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.frame_interval*args.test.num_frame, args.test.frame_interval).tolist()
    test_cam_ids = np.array(args.test.cam_ids).tolist()

    if args.test.cam_path is not None and args.test.pose_path is not None:
        # Check if this is a camera trajectory (list of cameras) or single camera
        try:
            with open(args.test.cam_path, 'r') as file:
                cam_data = json.load(file)
            
            if isinstance(cam_data, list):
                # Camera trajectory - multiple cameras
                cam_trajectory = []
                for cam_json in cam_data:
                    cam = {
                        'w2c': torch.as_tensor(np.array(cam_json['w2c']).reshape(4,4)).float().cuda(),
                        'K': torch.as_tensor(fovx_to_intrinsic(cam_json['fovx'] / 180 * np.pi, cam_json['height'], cam_json['width'])).cuda(),
                        'height': cam_json['height'],
                        'width': cam_json['width']
                    }
                    cam_trajectory.append(cam)
                print(f"Loaded camera trajectory with {len(cam_trajectory)} cameras")
            else:
                # Single camera
                cam_trajectory = [{
                    'w2c': torch.as_tensor(np.array(cam_data['w2c']).reshape(4,4)).float().cuda(),
                    'K': torch.as_tensor(fovx_to_intrinsic(cam_data['fovx'] / 180 * np.pi, cam_data['height'], cam_data['width'])).cuda(),
                    'height': cam_data['height'],
                    'width': cam_data['width']
                }]
                print("Loaded single camera")
        except Exception as e:
            print(f"Error loading camera data: {e}")
            return

        if 'smpl_params.npz' in args.test.pose_path:
            pose_list = load_thuman_pose_list(args.test.pose_path)
        elif args.test.pose_path.endswith('.json'):
            pose_list = load_json_pose_list(args.test.pose_path)
        else:
            pose_list = load_amass_pose_list(args.test.pose_path)

        if args.test.test_speed:
            testing_novel_cam_pose_speed(gaussians, args.out_dir, test_frame_ids, pose_list, cam_trajectory[0], background)
        else:
            if len(cam_trajectory) > 1:
                # Use 360-frame novel view generation for camera trajectories
                testing_360frame_novel_view(gaussians, args.out_dir, test_frame_ids, pose_list, cam_trajectory, background)
            else:
                # Use single camera mode
                testing_novel_cam_pose(gaussians, args.out_dir, test_frame_ids, pose_list, cam_trajectory[0], background)
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
    parser.add_argument('--num_frames', type=int, default=360, help='Number of frames to generate for 360-view')
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir, args.model_dir, args.test.cam_path, args.test.pose_path = pargs.data_dir, pargs.out_dir, pargs.model_dir, pargs.cam_path, pargs.pose_path
    args.test.is_test, args.test.test_speed = pargs.test, pargs.test_speed
    torch.backends.cuda.matmul.allow_tf32 = True

    testing(args)
