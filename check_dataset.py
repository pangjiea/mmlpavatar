#!/usr/bin/env python3
"""
检查SQ_02数据集的SMPL参数，验证是否包含expression和jaw_pose
"""

import numpy as np
import sys

def check_smpl_params():
    """检查SMPL参数文件"""
    print("检查SQ_02数据集的SMPL参数...")
    
    try:
        smpl_file = "/home/hello/data/SQ_02/smpl_params.npz"
        data = np.load(smpl_file, allow_pickle=True)
        
        print(f"SMPL参数文件包含的键: {list(data.keys())}")
        
        for key in data.keys():
            value = data[key]
            if hasattr(value, 'shape'):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)}")
                
        # 检查是否有expression和jaw_pose
        has_expression = 'expression' in data.keys()
        has_jaw_pose = 'jaw_pose' in data.keys()
        
        print(f"\n表情参数 (expression): {'✓ 存在' if has_expression else '✗ 不存在'}")
        print(f"下颚姿态 (jaw_pose): {'✓ 存在' if has_jaw_pose else '✗ 不存在'}")
        
        if has_expression:
            expression = data['expression']
            print(f"Expression shape: {expression.shape}")
            print(f"Expression sample: {expression[0] if len(expression) > 0 else 'empty'}")
            
        if has_jaw_pose:
            jaw_pose = data['jaw_pose']
            print(f"Jaw pose shape: {jaw_pose.shape}")
            print(f"Jaw pose sample: {jaw_pose[0] if len(jaw_pose) > 0 else 'empty'}")
            
        return has_expression, has_jaw_pose
        
    except Exception as e:
        print(f"错误: {e}")
        return False, False

if __name__ == "__main__":
    check_smpl_params()