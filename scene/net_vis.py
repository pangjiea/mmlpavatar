
IMAGE_ENCODE = 'torch'

import os
from os import path
import time
import torch
import sys
from tqdm import tqdm
import numpy as np
import json
import pickle
from scipy.spatial.transform import Rotation

from utils.image_utils import encode_bytes
from utils.net_utils import net_init, recv, send, is_connected
from scene.gaussian_model import GaussianModel

def get_render_choice(gaussians: GaussianModel, render_type, data):
    cam_pos = data['cam_pos']
    override_color = gaussians.get_color(cam_pos)
    return override_color 

def load_pose_list(file_path):
    with open(file_path, 'r') as file:
        pose_list = json.load(file)
    
    has_expression = False
    for pose in pose_list:
        for key in ['pose', 'Rh', 'Th']:
            pose[key] = np.array(pose[key]).astype(np.float32)
        # Handle expression if present, otherwise use default zeros
        if 'expression' in pose:
            pose['expression'] = np.array(pose['expression']).astype(np.float32)
            has_expression = True
        else:
            pose['expression'] = np.zeros(10, dtype=np.float32)
    
    if has_expression:
        print(f"✓ Successfully loaded expression parameters from {file_path}")
        print(f"  - Total frames with expression: {len(pose_list)}")
        print(f"  - Expression dimension: {pose_list[0]['expression'].shape[0]}")
    else:
        print(f"⚠ No expression parameters found in {file_path}, using default zeros")
    
    return pose_list

def load_cam_list(file_path):
    with open(file_path, 'r') as file:
        cam_list = json.load(file)
    for cam in cam_list:
        for key in ['K', 'w2c']:
            cam[key] = np.array(cam[key]).astype(np.float32)
    return cam_list

def load_model(model_dir):
    ckpt_path = path.join(model_dir, sorted([pth for pth in os.listdir(model_dir) if 'chkpnt' in pth and '.pth' in pth], key=lambda s: (len(s), s))[-1])
    gaussians = GaussianModel()
    load_data = torch.load(ckpt_path, weights_only=False)
    first_iter = load_data['iteration']
    print(f'Loading checkpoint from ITER {first_iter}')
    gaussians.restore(load_data)
    return gaussians

class Visualizer:
    gaussians: GaussianModel
    cam_list = None
    pose_list = None
    is_send_initial_data = False
    def __init__(self, in_training=False):
        self.in_training = in_training
    
    def net_init(self, ip, port):
        net_init(ip, port)

    def load_model(self, model_dir):
        self.gaussians = load_model(model_dir)
        self.load_cams_poses(model_dir)

    def load_cams_poses(self, model_dir):
        self.cam_list = load_cam_list(path.join(model_dir, 'cameras.json'))
        self.pose_list = load_pose_list(path.join(model_dir, 'poses.json'))
        
        # Print summary of loaded data
        print(f"\n=== Model Loading Summary ===")
        print(f"Model directory: {model_dir}")
        print(f"Cameras loaded: {len(self.cam_list)}")
        print(f"Poses loaded: {len(self.pose_list)}")
        
        # Check if expressions are present in loaded poses
        has_expressions = any('expression' in pose and not np.allclose(pose['expression'], 0) for pose in self.pose_list)
        if has_expressions:
            print(f"✓ Expression data: Available")
            # Print expression range for first few frames
            for i in range(min(3, len(self.pose_list))):
                expr = self.pose_list[i]['expression']
                print(f"  Frame {i}: expression range [{expr.min():.3f}, {expr.max():.3f}]")
        else:
            print(f"⚠ Expression data: Not available (using default zeros)")
        print(f"============================\n")

    @staticmethod
    def pklbytes_to_data(pklbytes):
        if pklbytes is None: return None 
        data = pickle.loads(pklbytes)
        data['cam_pos'] = np.linalg.inv(data['w2c'])[:3,3]

        cuda_keys = ['K', 'w2c', 'background', 'cam_pos']
        cpu_keys = ['pose', 'Rh', 'Th', 'expression']
        for key in cuda_keys:
            data[key] = torch.from_numpy(data[key]).float().cuda(non_blocking=True)
        for key in cpu_keys:
            data[key] = torch.from_numpy(data[key]).float()
        
        return data

    @torch.no_grad()
    def visualizing(self):
        if not is_connected(): return False

        data = self.pklbytes_to_data(recv())
        if data is None: return False

        gaussians = self.gaussians   
        if not self.in_training and 'is_test' in data:
            gaussians.is_test = data['is_test']
        if 'is_gsparam_bs' in data:
            gaussians.is_gsparam_bs = data['is_gsparam_bs']
        if 'is_dxyz_bs' in data:
            gaussians.is_dxyz_bs = data['is_dxyz_bs']
        if 'sh_degree' in data:
            sh_degree = max(0, min(data['sh_degree'], 1))
            gaussians.sh_degree = sh_degree

        gaussians.Rh, gaussians.Th = data['Rh'], data['Th']
        gaussians.smpl_poses = data['pose']
        # Handle expression parameter
        if 'expression' in data:
            gaussians.expression = data['expression']
            # Print expression info for first frame or when expression changes significantly
            if not hasattr(self, '_last_expression_print') or torch.norm(data['expression'] - self._last_expression_print) > 0.1:
                expr_range = [data['expression'].min().item(), data['expression'].max().item()]
                print(f"🎭 Using expression parameters: range [{expr_range[0]:.3f}, {expr_range[1]:.3f}]")
                self._last_expression_print = data['expression'].clone()
        else:
            gaussians.expression = torch.zeros(10, dtype=torch.float32)
            if not hasattr(self, '_expression_warning_printed'):
                print(f"⚠ No expression parameters received, using neutral expression")
                self._expression_warning_printed = True

        override_color = get_render_choice(gaussians, None, data)

        image, _, _ = gaussians.render(
            cam=data, 
            override_color=override_color, 
            scaling_modifier=data['scaling_modifier'], 
            background=data['background'],
        )
        image = (torch.clamp(image, min=0, max=1.0) * 255).byte()
        if IMAGE_ENCODE != 'torch': image = image.contiguous().cpu().numpy()
        image_bytes = encode_bytes(image, IMAGE_ENCODE)
        
        ret_data = {}

        ret_data['image_bytes'] = image_bytes
        ret_data['gaussian_num'] = len(gaussians._xyz)
        
        if self.is_send_initial_data:
            self.is_send_initial_data = False
            ret_data['camera_list'] = self.cam_list
            ret_data['pose_list'] = self.pose_list

        send(pickle.dumps(ret_data))
        return True

