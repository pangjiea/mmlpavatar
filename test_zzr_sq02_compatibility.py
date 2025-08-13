#!/usr/bin/env python3
"""
Test script to verify zzr_sq02_action_sequence_360frames.json works with z1-z7 camera trajectory
"""

import json
import numpy as np

def load_action_sequence(action_path):
    """Load action sequence"""
    print(f"Loading action sequence from: {action_path}")
    with open(action_path, 'r') as f:
        action_data = json.load(f)
    
    print(f"Action sequence loaded with {len(action_data)} frames")
    
    # Check structure of first frame
    first_frame = action_data[0]
    print(f"First frame keys: {list(first_frame.keys())}")
    print(f"Pose dimensions: {len(first_frame['pose'])}")
    print(f"Rh dimensions: {len(first_frame['Rh'])}")
    print(f"Th dimensions: {len(first_frame['Th'])}")
    
    return action_data

def load_camera_trajectory(camera_path):
    """Load camera trajectory"""
    print(f"Loading camera trajectory from: {camera_path}")
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)
    
    print(f"Camera trajectory loaded with {len(camera_data)} frames")
    
    # Check structure of first frame
    first_frame = camera_data[0]
    print(f"First camera frame keys: {list(first_frame.keys())}")
    
    return camera_data

def verify_compatibility(action_data, camera_data):
    """Verify that action sequence and camera trajectory are compatible"""
    print("\n=== Compatibility Check ===")
    
    action_frames = len(action_data)
    camera_frames = len(camera_data)
    
    print(f"Action sequence frames: {action_frames}")
    print(f"Camera trajectory frames: {camera_frames}")
    
    if action_frames == camera_frames:
        print("✓ Frame count matches!")
    else:
        print(f"✗ Frame count mismatch: {action_frames} vs {camera_frames}")
    
    # Check pose data
    first_pose = action_data[0]['pose']
    print(f"Pose data dimensions: {len(first_pose)}")
    
    if len(first_pose) == 165:
        print("✓ Pose data has correct dimensions (165)")
    else:
        print(f"✗ Pose data has incorrect dimensions: {len(first_pose)}")
    
    # Check camera data
    first_camera = camera_data[0]
    if 'w2c' in first_camera and 'K' in first_camera:
        print("✓ Camera data has required fields (w2c, K)")
    else:
        print("✗ Camera data missing required fields")
    
    return action_frames == camera_frames

def main():
    # Paths
    action_path = "/home/hello/code/mmlphuman/zzr_sq02_action_sequence_360frames.json"
    camera_path = "/home/hello/code/mmlphuman/z1_to_z7_simple_trajectory.json"
    
    # Load data
    action_data = load_action_sequence(action_path)
    camera_data = load_camera_trajectory(camera_path)
    
    # Verify compatibility
    is_compatible = verify_compatibility(action_data, camera_data)
    
    if is_compatible:
        print("\n✓ SUCCESS: Action sequence is compatible with camera trajectory!")
        print("Ready for video generation with:")
        print("- zzr poses")
        print("- sq02 Rh and Th parameters")
        print("- z1-z7 camera trajectory (360 frames, 12 seconds)")
    else:
        print("\n✗ FAILURE: Compatibility issues found")

if __name__ == "__main__":
    main()