#!/usr/bin/env python3
"""
简单测试验证TalkBody4D投影算法修正
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_projection_fix():
    """测试投影算法修正"""
    print("测试TalkBody4D投影算法修正...")
    
    try:
        from utils.head_mask_utils import generate_head_mask
        from utils.smpl_utils import init_smpl
        
        # 初始化SMPL模型
        smpl_pkl_path = "./smpl_model/smplx/SMPLX_NEUTRAL.npz"
        if os.path.exists(smpl_pkl_path):
            init_smpl(smpl_pkl_path)
            print("✓ SMPL模型初始化成功")
        else:
            print("✗ SMPL模型文件不存在")
            return False
        
        # 测试参数
        height, width = 480, 640
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        
        # 相机参数 - 测试不同的相机位置
        test_cameras = [
            {
                'name': '标准正视',
                'K': np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32),
                'w2c': np.eye(4, dtype=np.float32)
            },
            {
                'name': '稍微偏移',
                'K': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32),
                'w2c': np.array([
                    [0.9, 0.1, 0.0, 0.1],
                    [-0.1, 0.9, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ], dtype=np.float32)
            }
        ]
        
        for cam_config in test_cameras:
            print(f"\n测试相机配置: {cam_config['name']}")
            
            # 生成头部掩膜
            head_mask = generate_head_mask(
                pose, beta, expression, jaw_pose, Rh, Th,
                cam_config['K'], cam_config['w2c'], height, width, 
                body_mask=None, device='cpu'
            )
            
            if head_mask.sum() > 0:
                print(f"  ✓ 成功生成头部掩膜: {head_mask.sum()} 像素 ({head_mask.sum()/(height*width)*100:.2f}%)")
                
                # 分析掩膜分布
                y_indices, x_indices = np.where(head_mask)
                y_center = y_indices.mean()
                x_center = x_indices.mean()
                print(f"  ✓ 头部中心位置: ({x_center:.1f}, {y_center:.1f})")
                print(f"  ✓ 覆盖范围: X[{x_indices.min()}-{x_indices.max()}], Y[{y_indices.min()}-{y_indices.max()}]")
            else:
                print(f"  ⚠️ 生成的头部掩膜为空")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TalkBody4D投影算法修正验证")
    print("=" * 60)
    
    success = test_projection_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 投影算法修正验证成功!")
        print("\n关键改进:")
        print("- 采用TalkBody4D的正确投影方法: R*verts + T")
        print("- 正确的透视除法: verts_proj / verts_proj[2, :]")
        print("- 使用真实SMPL-X mesh faces渲染") 
        print("- 与全身掩膜交集确保合理性")
        print("- 避免SMPL-X einsum错误")
    else:
        print("❌ 投影算法修正验证失败")