#!/usr/bin/env python3
"""
Script to create 360-frame action sequence with zzr poses and sq02 parameters for z1-z7 camera trajectory
"""

import numpy as np
import json
import os
import glob

def load_zzr_poses(smpl_path):
    """Load SMPL parameters from zzr dataset"""
    print(f"Loading zzr SMPL parameters from: {smpl_path}")
    smpl_data = np.load(smpl_path, allow_pickle=True)
    
    # Extract pose parameters from zzr
    global_orient = smpl_data['global_orient']      # (N, 3)
    body_pose = smpl_data['body_pose']              # (N, 63)
    transl = smpl_data['transl']                    # (N, 3)
    left_hand_pose = smpl_data['left_hand_pose']    # (N, 45)
    right_hand_pose = smpl_data['right_hand_pose']  # (N, 45)
    
    print(f"zzr poses shape: global_orient={global_orient.shape}, body_pose={body_pose.shape}, transl={transl.shape}")
    print(f"hand poses: left_hand_pose={left_hand_pose.shape}, right_hand_pose={right_hand_pose.shape}")
    
    # Create full pose in the expected format (165 dimensions)
    poses = []
    N = len(global_orient)
    for frame_id in range(N):
        pose = np.concatenate([
            global_orient[frame_id],           # 3
            body_pose[frame_id],               # 63
            np.zeros(3, dtype=np.float32),     # 3 (jaw)
            np.zeros(6, dtype=np.float32),     # 6 (expression)
            left_hand_pose[frame_id],          # 45
            right_hand_pose[frame_id],         # 45
        ], axis=0)  # Total: 3+63+3+6+45+45 = 165
        
        poses.append(pose)
    
    poses = np.array(poses)  # (N, 165)
    
    return poses, transl

def load_sq_02_params(json_dir, max_frames=360):
    """Load SQ_02 parameters from JSON files"""
    print(f"Loading SQ_02 parameters from: {json_dir}")
    
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))[:max_frames]
    sq02_rh = []
    sq02_th = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract Rh and Th from translation (assuming transl is [x, y, z])
        transl = np.array(data['transl'][0])  # (3,)
        th = transl  # Th is the translation
        rh = np.array([0.0, 0.0, 0.0])  # Rh is rotation, assuming no rotation for now
        
        sq02_rh.append(rh)
        sq02_th.append(th)
    
    sq02_rh = np.array(sq02_rh)        # (N, 3)
    sq02_th = np.array(sq02_th)        # (N, 3)
    
    print(f"SQ_02 parameters loaded: Rh={sq02_rh.shape}, Th={sq02_th.shape}")
    
    return sq02_rh, sq02_th

def create_zzr_sq02_sequence_360(zzr_poses, sq02_rh, sq02_th, num_frames=361):
    """Create 361-frame sequence with zzr poses and SQ_02 other parameters"""
    
    print(f"Creating {num_frames}-frame sequence with zzr poses and sq02 parameters")
    
    # Cycle zzr poses to match exactly 361 frames
    zzr_frames = len(zzr_poses)
    final_poses = []
    
    for i in range(num_frames):
        frame_idx = i % zzr_frames
        final_poses.append(zzr_poses[frame_idx])
    
    final_poses = np.array(final_poses)  # (361, 165)
    
    # Cycle SQ_02 Rh and Th to match exactly 361 frames
    final_rh = []
    final_th = []
    
    for i in range(num_frames):
        frame_idx = i % len(sq02_rh)
        final_rh.append(sq02_rh[frame_idx])
        final_th.append(sq02_th[frame_idx])
    
    final_rh = np.array(final_rh)  # (361, 3)
    final_th = np.array(final_th)  # (361, 3)
    
    print(f"Final sequence: poses={final_poses.shape}, Rh={final_rh.shape}, Th={final_th.shape}")
    
    return final_poses, final_rh, final_th

def save_action_sequence(poses, rh, th, output_path):
    """Save action sequence to JSON file"""
    print(f"Saving action sequence to: {output_path}")
    
    sequence_data = []
    for i in range(len(poses)):
        frame_data = {
            'pose': poses[i].tolist(),
            'Rh': rh[i].tolist(),
            'Th': th[i].tolist()
        }
        sequence_data.append(frame_data)
    
    with open(output_path, 'w') as f:
        json.dump(sequence_data, f, indent=2)
    
    print(f"Action sequence saved with {len(sequence_data)} frames")

def main():
    # Paths
    zzr_smpl_path = "/home/hello/data/avatarrex_zzr/smpl_params.npz"
    sq02_json_dir = "/home/hello/data/SQ_02/smplx_fitting"
    output_path = "/home/hello/code/mmlphuman/zzr_sq02_action_sequence_360frames.json"
    
    # Load data
    zzr_poses, zzr_transl = load_zzr_poses(zzr_smpl_path)
    sq02_rh, sq02_th = load_sq_02_params(sq02_json_dir, max_frames=360)
    
    # Create 361-frame action sequence with zzr poses and sq02 parameters
    final_poses, final_rh, final_th = create_zzr_sq02_sequence_360(
        zzr_poses, sq02_rh, sq02_th, num_frames=361
    )
    
    # Save action sequence
    save_action_sequence(final_poses, final_rh, final_th, output_path)
    
    print("361-frame action sequence creation completed!")
    print(f"Output saved to: {output_path}")
    print("This sequence uses zzr poses with sq02 Rh and Th parameters")
    print("Perfect for z1-z7 camera trajectory (361 frames, ~12 seconds)")

if __name__ == "__main__":
    main()