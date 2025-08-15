#!/usr/bin/env python3
"""
Script to replace SQ02 body and hand poses with ZZR poses
Keep all other SQ02 parameters including Rh, Th
"""

import numpy as np
import pickle
import sys
import os

def load_smpl_params(npz_path):
    """Load SMPL parameters from npz file"""
    data = np.load(npz_path, allow_pickle=True)
    return dict(data)

def analyze_params(data, name):
    """Analyze SMPL parameters structure"""
    print(f"\n=== {name} Data Analysis ===")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"{name} {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"{name} {key}: type={type(data[key])}")

def create_replaced_action(zzr_data, sq02_data, output_path):
    """
    Create replaced action sequence:
    - Start with SQ02 data as base
    - Replace only: body_pose, left_hand_pose, right_hand_pose with ZZR data
    - Keep ALL other SQ02 parameters including Rh, Th, betas, etc.
    """
    
    print(f"\nCreating replaced action sequence (ZZR poses -> SQ02 base)...")
    
    # Get frame counts
    zzr_frames = len(zzr_data['body_pose'])
    sq02_frames = len(sq02_data['body_pose'])
    min_frames = min(zzr_frames, sq02_frames)
    
    print(f"ZZR frames: {zzr_frames}, SQ02 frames: {sq02_frames}, Using: {min_frames}")
    
    # Start with SQ02 data as base
    replaced_data = {}
    for key in sq02_data.keys():
        if isinstance(sq02_data[key], np.ndarray) and len(sq02_data[key]) > min_frames:
            replaced_data[key] = sq02_data[key][:min_frames]
        else:
            replaced_data[key] = sq02_data[key]
    
    # Replace only the pose parameters from ZZR
    replaced_data['body_pose'] = zzr_data['body_pose'][:min_frames]
    replaced_data['left_hand_pose'] = zzr_data['left_hand_pose'][:min_frames]
    replaced_data['right_hand_pose'] = zzr_data['right_hand_pose'][:min_frames]
    
    # Verify all arrays have same length
    for key in replaced_data.keys():
        if isinstance(replaced_data[key], np.ndarray) and len(replaced_data[key]) != min_frames:
            print(f"Warning: {key} has length {len(replaced_data[key])}, expected {min_frames}")
    
    # Save replaced data
    np.savez(output_path, **replaced_data)
    print(f"Replaced action saved to: {output_path}")
    
    return replaced_data, min_frames

def verify_replaced_data(replaced_data, zzr_data, sq02_data, min_frames):
    """Verify which parameters come from which source"""
    print(f"\n=== Parameter Source Verification ===")
    print(f"Total frames: {min_frames}")
    
    replaced_params = ['body_pose', 'left_hand_pose', 'right_hand_pose']
    kept_params = [key for key in sq02_data.keys() if key not in replaced_params]
    
    print(f"\nReplaced with ZZR:")
    for param in replaced_params:
        if param in replaced_data:
            print(f"  {param}: shape={replaced_data[param].shape}")
    
    print(f"\nKept from SQ02:")
    for param in kept_params:
        if param in replaced_data:
            print(f"  {param}: shape={replaced_data[param].shape}")
    
    # Verify first few values match sources
    print(f"\n=== Verification ===")
    for param in replaced_params:
        if param in replaced_data and param in zzr_data:
            match = np.allclose(replaced_data[param][0], zzr_data[param][0])
            print(f"{param} matches ZZR: {match}")
    
    for param in kept_params:
        if param in replaced_data and param in sq02_data:
            if isinstance(replaced_data[param], np.ndarray) and isinstance(sq02_data[param], np.ndarray):
                if len(replaced_data[param]) > 0 and len(sq02_data[param]) > 0:
                    match = np.allclose(replaced_data[param][0], sq02_data[param][0])
                    print(f"{param} matches SQ02: {match}")

def main():
    # Paths
    zzr_path = '/home/hello/data/avatarrex_zzr/smpl_params.npz'
    sq02_path = '/home/hello/data/SQ_02/smpl_params.npz'
    output_path = '/home/hello/data/replaced_zzr_sq02_action.npz'
    
    print("Loading SMPL parameters...")
    
    # Load data
    zzr_data = load_smpl_params(zzr_path)
    sq02_data = load_smpl_params(sq02_path)
    
    # Analyze structure
    analyze_params(zzr_data, "ZZR")
    analyze_params(sq02_data, "SQ02")
    
    # Create replaced action
    replaced_data, min_frames = create_replaced_action(zzr_data, sq02_data, output_path)
    
    # Verify
    verify_replaced_data(replaced_data, zzr_data, sq02_data, min_frames)
    
    print(f"\n=== Summary ===")
    print(f"Replaced action sequence created successfully!")
    print(f"Output file: {output_path}")
    print(f"Total frames: {min_frames}")
    print(f"Replaced with ZZR: body_pose, left_hand_pose, right_hand_pose")
    print(f"Kept from SQ02: ALL other parameters including Rh, Th, betas, expression, jaw_pose, leye_pose, reye_pose, global_orient, transl, v_shape")

if __name__ == "__main__":
    main()