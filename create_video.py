#!/usr/bin/env python3

import os
import argparse
import imageio.v2 as imageio
import numpy as np
from pathlib import Path

def create_video_from_frames(frame_dir, output_path, fps=30, pattern="*.png"):
    """
    Create a video from a directory of frame images
    
    Args:
        frame_dir: Directory containing frame images
        output_path: Output video file path
        fps: Frames per second for the video
        pattern: File pattern to match (default: *.png)
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    
    # Get all frame files matching the pattern
    frame_files = sorted(frame_dir.glob(pattern))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frame_dir} with pattern {pattern}")
    
    print(f"Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame = imageio.imread(frame_files[0])
    height, width = first_frame.shape[:2]
    
    print(f"Frame dimensions: {width}x{height}")
    print(f"Creating video at {fps} FPS...")
    
    # Create video writer
    writer = imageio.get_writer(output_path, fps=fps)
    
    # Write frames
    for frame_file in frame_files:
        frame = imageio.imread(frame_file)
        writer.append_data(frame)
    
    writer.close()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create video from rendered frames")
    parser.add_argument('--frame_dir', type=str, required=True,
                       help='Directory containing frame images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video file path')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='File pattern to match (default: *.png)')
    
    args = parser.parse_args()
    
    create_video_from_frames(args.frame_dir, args.output, args.fps, args.pattern)

if __name__ == "__main__":
    main()