#!/usr/bin/env python3
"""
Script to create simple z1-z7 camera trajectory for 12 seconds (360 frames)
"""

import json
import numpy as np

def load_z_cameras(z_cameras_path):
    """Load z camera positions"""
    with open(z_cameras_path, 'r') as f:
        z_cameras = json.load(f)
    return z_cameras

def create_simple_z1_to_z7_trajectory(z_cameras, num_frames=360):
    """Create simple camera trajectory from z1 to z7 in 12 seconds"""
    
    # Extract the 7 camera positions (z1 to z7)
    cameras = z_cameras  # Should have 7 cameras
    
    print(f"Loaded {len(cameras)} z cameras")
    
    # Create simple interpolation from z1 to z7
    # We need to distribute 360 frames across 7 camera positions
    frames_per_camera = num_frames // (len(cameras) - 1)
    remainder_frames = num_frames % (len(cameras) - 1)
    
    print(f"Frames per camera segment: {frames_per_camera}")
    print(f"Remainder frames: {remainder_frames}")
    
    trajectory = []
    
    # Interpolate between consecutive cameras
    for i in range(len(cameras) - 1):
        start_cam = cameras[i]
        end_cam = cameras[i + 1]
        
        # Calculate frames for this segment
        segment_frames = frames_per_camera
        if i < remainder_frames:
            segment_frames += 1
        
        print(f"Camera {i+1} to {i+2}: {segment_frames} frames")
        
        # Interpolate between start and end camera
        for frame in range(segment_frames):
            t = frame / segment_frames if segment_frames > 1 else 0
            
            # Interpolate w2c matrix
            interpolated_w2c = []
            for j in range(16):
                start_val = start_cam['w2c'][j]
                end_val = end_cam['w2c'][j]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated_w2c.append(interpolated_val)
            
            # Interpolate K matrix (camera intrinsics)
            interpolated_K = []
            for j in range(9):
                start_val = start_cam['K'][j]
                end_val = end_cam['K'][j]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated_K.append(interpolated_val)
            
            # Create interpolated camera
            interpolated_cam = {
                'w2c': interpolated_w2c,
                'K': interpolated_K,
                'fovx': start_cam['fovx'],
                'height': start_cam['height'],
                'width': start_cam['width']
            }
            
            trajectory.append(interpolated_cam)
    
    # Add the final camera (z7)
    trajectory.append(cameras[-1])
    
    print(f"Created trajectory with {len(trajectory)} frames")
    
    return trajectory

def save_camera_trajectory(trajectory, output_path):
    """Save camera trajectory to JSON file"""
    print(f"Saving camera trajectory to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    
    print(f"Camera trajectory saved with {len(trajectory)} frames")

def main():
    # Paths
    z_cameras_path = "/home/hello/code/mmlphuman/z_cameras_trajectory_simple.json"
    output_path = "/home/hello/code/mmlphuman/z1_to_z7_simple_trajectory.json"
    
    # Load z cameras
    z_cameras = load_z_cameras(z_cameras_path)
    
    # Create simple z1 to z7 trajectory
    trajectory = create_simple_z1_to_z7_trajectory(z_cameras, num_frames=360)
    
    # Save trajectory
    save_camera_trajectory(trajectory, output_path)
    
    print("Simple camera trajectory creation completed!")
    print(f"Output saved to: {output_path}")
    print("This trajectory smoothly moves from z1 to z7 in 12 seconds")

if __name__ == "__main__":
    main()