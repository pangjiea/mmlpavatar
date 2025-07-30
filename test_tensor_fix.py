#!/usr/bin/env python3
"""
Test the tensor dimension fix in the _get_joint_features_for_type method
"""

import torch
import numpy as np
import sys
import os

# Add project path
sys.path.append('/home/hello/code/mmlphuman')

def test_tensor_dimensions():
    """Test tensor dimension handling in GaussianModel"""
    print("Testing tensor dimension fix...")
    
    try:
        from scene.gaussian_model import GaussianModel
        from utils.smpl_utils import init_smpl_pose
        
        # Initialize SMPL
        init_smpl_pose()
        
        # Create GaussianModel instance
        gaussians = GaussianModel()
        
        # Simulate the problematic scenario - 2D tensors for expression and jaw_pose
        gaussians.expression = torch.zeros(1, 10, dtype=torch.float32)  # 2D tensor [1, 10]
        gaussians.jaw_pose = torch.zeros(1, 3, dtype=torch.float32)     # 2D tensor [1, 3]
        gaussians.smpl_poses_cuda = torch.zeros(165, dtype=torch.float32).cuda()  # SMPL poses
        gaussians.is_test = False
        
        print(f"✓ Expression shape: {gaussians.expression.shape}")
        print(f"✓ Jaw pose shape: {gaussians.jaw_pose.shape}")
        
        # Test different feature types
        print("\nTesting feature extraction...")
        
        # Test 'all' features (this was causing the error)
        try:
            features_all = gaussians._get_joint_features_for_type('all')
            print(f"✓ All features shape: {features_all.shape}")
            print(f"✓ All features device: {features_all.device}")
        except Exception as e:
            print(f"✗ All features failed: {e}")
            return False
        
        # Test 'head' features  
        try:
            features_head = gaussians._get_joint_features_for_type('head')
            print(f"✓ Head features shape: {features_head.shape}")
            print(f"✓ Head features device: {features_head.device}")
        except Exception as e:
            print(f"✗ Head features failed: {e}")
            return False
            
        # Test 'body' features
        try:
            features_body = gaussians._get_joint_features_for_type('body')
            print(f"✓ Body features shape: {features_body.shape}")
            print(f"✓ Body features device: {features_body.device}")
        except Exception as e:
            print(f"✗ Body features failed: {e}")
            return False
        
        # Verify expected dimensions
        expected_all = 63 + 10 + 3  # body + expression + jaw = 76
        expected_head = 10 + 3 + 3  # expression + jaw + neck = 16
        expected_body = 63 + 3      # body + neck = 66
        
        if features_all.shape[0] == expected_all:
            print(f"✓ All features dimension correct: {expected_all}")
        else:
            print(f"✗ All features dimension wrong: expected {expected_all}, got {features_all.shape[0]}")
            return False
            
        if features_head.shape[0] == expected_head:
            print(f"✓ Head features dimension correct: {expected_head}")
        else:
            print(f"✗ Head features dimension wrong: expected {expected_head}, got {features_head.shape[0]}")
            return False
            
        if features_body.shape[0] == expected_body:
            print(f"✓ Body features dimension correct: {expected_body}")
        else:
            print(f"✗ Body features dimension wrong: expected {expected_body}, got {features_body.shape[0]}")
            return False
        
        print("\n🎉 All tensor dimension tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Tensor Dimension Fix Test")
    print("=" * 60)
    
    success = test_tensor_dimensions()
    
    if success:
        print("\n✅ Tensor dimension fix is working correctly!")
        print("The error at iteration 2000 should be resolved.")
    else:
        print("\n❌ Tensor dimension fix needs more work.")