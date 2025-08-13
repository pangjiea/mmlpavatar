#!/usr/bin/env python3

import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import os

def load_camera_json(json_path):
    """Load camera parameters from JSON file"""
    with open(json_path, 'r') as f:
        cam_data = json.load(f)
    
    w2c = np.array(cam_data['w2c']).reshape(4, 4)
    K = np.array(cam_data['K']).reshape(3, 3)
    fovx = cam_data['fovx']
    height = cam_data['height']
    width = cam_data['width']
    
    return {
        'w2c': w2c,
        'K': K,
        'fovx': fovx,
        'height': height,
        'width': width
    }

def w2c_to_c2w(w2c):
    """Convert world-to-camera matrix to camera-to-world matrix"""
    return np.linalg.inv(w2c)

def c2w_to_w2c(c2w):
    """Convert camera-to-world matrix to world-to-camera matrix"""
    return np.linalg.inv(c2w)

def slerp_cameras(cam1, cam2, t):
    """Spherical linear interpolation between two cameras"""
    # Convert to camera-to-world matrices
    c2w1 = w2c_to_c2w(cam1['w2c'])
    c2w2 = w2c_to_c2w(cam2['w2c'])
    
    # Extract rotation and translation
    R1 = c2w1[:3, :3]
    R2 = c2w2[:3, :3]
    t1 = c2w1[:3, 3]
    t2 = c2w2[:3, 3]
    
    # Slerp rotation
    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)
    slerp = Slerp([0, 1], Rotation.concatenate([rot1, rot2]))
    R_interp = slerp(t).as_matrix()
    
    # Linear interpolation for translation
    t_interp = (1 - t) * t1 + t * t2
    
    # Reconstruct camera-to-world matrix
    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = R_interp
    c2w_interp[:3, 3] = t_interp
    
    # Convert back to world-to-camera
    w2c_interp = c2w_to_w2c(c2w_interp)
    
    # Interpolate intrinsics
    K_interp = (1 - t) * cam1['K'] + t * cam2['K']
    fovx_interp = (1 - t) * cam1['fovx'] + t * cam2['fovx']
    
    return {
        'w2c': w2c_interp,
        'K': K_interp,
        'fovx': fovx_interp,
        'height': cam1['height'],
        'width': cam1['width']
    }

def create_circular_camera_path(cam_files, num_frames=360, smoothness=1.0):
    """
    Create a circular camera path with smooth interpolation
    
    Args:
        cam_files: List of camera JSON files [cam1.json, cam2.json, cam3.json, cam4.json, cam5.json]
        num_frames: Total number of frames to generate
        smoothness: Smoothing factor for interpolation
    """
    # Load all cameras
    cameras = [load_camera_json(cam_file) for cam_file in cam_files]
    
    # Create path: 1->2->3->4->5->4->3->2->1 (smooth circular)
    path_points = []
    
    # Forward path: 1->2->3->4->5
    for i in range(len(cameras)):
        path_points.append(cameras[i])
    
    # Backward path: 5->4->3->2->1
    for i in range(len(cameras)-2, 0, -1):
        path_points.append(cameras[i])
    
    # Generate interpolated frames
    interpolated_cameras = []
    
    # Calculate total segments
    total_segments = len(path_points) - 1
    frames_per_segment = num_frames // total_segments
    
    for segment_idx in range(total_segments):
        start_cam = path_points[segment_idx]
        end_cam = path_points[segment_idx + 1]
        
        for frame_idx in range(frames_per_segment):
            if len(interpolated_cameras) >= num_frames:
                break
                
            t = frame_idx / frames_per_segment
            # Apply smooth easing
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            
            interp_cam = slerp_cameras(start_cam, end_cam, t_smooth)
            interpolated_cameras.append(interp_cam)
    
    # Ensure we have exactly num_frames
    while len(interpolated_cameras) < num_frames:
        interpolated_cameras.append(interpolated_cameras[-1])
    
    return interpolated_cameras[:num_frames]

def save_camera_trajectory(cameras, output_path):
    """Save camera trajectory to JSON file"""
    trajectory_data = []
    
    for i, cam in enumerate(cameras):
        cam_data = {
            'w2c': cam['w2c'].flatten().tolist(),
            'K': cam['K'].flatten().tolist(),
            'fovx': cam['fovx'],
            'height': cam['height'],
            'width': cam['width']
        }
        trajectory_data.append(cam_data)
    
    with open(output_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate circular camera trajectory')
    parser.add_argument('--cam_files', nargs='+', 
                       default=['/tmp/cam1.json', '/tmp/cam2.json', '/tmp/cam3.json', 
                               '/tmp/cam4.json', '/tmp/cam5.json'],
                       help='Camera JSON files')
    parser.add_argument('--num_frames', type=int, default=360,
                       help='Number of frames to generate')
    parser.add_argument('--output', type=str, default='./output/camera_trajectory_360frames.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    print(f"Loading {len(args.cam_files)} camera files...")
    print(f"Generating {args.num_frames} frames...")
    
    # Generate camera trajectory
    cameras = create_circular_camera_path(args.cam_files, args.num_frames)
    
    # Save trajectory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_camera_trajectory(cameras, args.output)
    
    print(f"Camera trajectory saved to {args.output}")
    print(f"Generated {len(cameras)} camera poses")

if __name__ == "__main__":
    main()