#!/usr/bin/env python3
"""
测试头部身体分离功能的训练集成 - 简化版本用于验证
"""

import torch
import numpy as np
import os
import sys
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_training_functionality():
    """测试训练功能"""
    print("开始测试训练功能...")
    
    try:
        # 导入必要模块
        from scene.gaussian_model import GaussianModel
        from utils.loss_utils import head_body_separation_loss, head_consistency_loss
        from utils.smpl_utils import init_smpl_pose
        
        print("✓ 模块导入成功")
        
        # 初始化SMPL
        init_smpl_pose()
        print("✓ SMPL初始化成功")
        
        # 创建GaussianModel
        gaussians = GaussianModel()
        print("✓ GaussianModel创建成功")
        
        # 测试基本属性
        required_attrs = [
            'head_encoder_params', 'body_encoder_params',
            'head_gs_mask', 'body_gs_mask', 'head_face_indices'
        ]
        
        for attr in required_attrs:
            assert hasattr(gaussians, attr), f"缺少属性: {attr}"
        print("✓ 所有必要属性存在")
        
        # 模拟训练数据
        gaussians.expression = torch.zeros(10, dtype=torch.float32)
        gaussians.jaw_pose = torch.zeros(3, dtype=torch.float32)
        gaussians.smpl_poses_cuda = torch.zeros(165, dtype=torch.float32, device='cuda')
        
        print("✓ 训练数据设置成功")
        
        # 测试损失计算 (在没有head-body separation初始化时应该返回0)
        separation_loss, loss_dict = head_body_separation_loss(gaussians)
        print(f"✓ 分离损失计算成功: {separation_loss.item():.6f}")
        
        # 测试head mask一致性损失
        head_mask_gt = torch.zeros(128, 128, device='cuda')
        consistency_loss = head_consistency_loss(gaussians, head_mask_gt)
        print(f"✓ 一致性损失计算成功: {consistency_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """测试数据集加载功能"""
    print("\n开始测试数据集加载...")
    
    try:
        from scene.dataset import get_dataset_type
        
        # 测试数据集类型检测
        assert callable(get_dataset_type), "get_dataset_type应该可调用"
        print("✓ 数据集类型检测功能可用")
        
        # 测试头部掩膜工具
        from utils.head_mask_utils import get_neck_pose_from_full_pose
        
        # 测试颈部姿态提取
        full_pose = torch.zeros(165, dtype=torch.float32)
        neck_pose = get_neck_pose_from_full_pose(full_pose)
        assert neck_pose.shape == (3,), f"颈部姿态应该是3维，得到{neck_pose.shape}"
        print(f"✓ 颈部姿态提取成功: {neck_pose.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """测试配置集成"""
    print("\n开始测试配置集成...")
    
    try:
        # 测试SQ_02配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        if os.path.exists(config_path):
            args = OmegaConf.load(config_path)
            
            # 检查head-body separation参数
            required_params = [
                'lambda_head_body_separation',
                'lambda_head_body_feature_diff',
                'lambda_head_body_spatial_sep',
                'lambda_head_consistency', 
                'iteration_head_body_loss'
            ]
            
            for param in required_params:
                if hasattr(args, param):
                    print(f"✓ {param}: {getattr(args, param)}")
                else:
                    print(f"✗ 缺少参数: {param}")
                    return False
                    
            print("✓ SQ_02配置验证成功")
            return True
        else:
            print(f"✗ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ 配置集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """测试训练脚本导入"""
    print("\n开始测试训练脚本...")
    
    try:
        # 测试是否能导入训练脚本的关键部分
        from utils.loss_utils import l1_loss, psnr, lpips_loss, dxyz_smooth_loss, gaussian_scaling_loss
        from utils.loss_utils import head_body_separation_loss, head_consistency_loss
        
        print("✓ 所有损失函数导入成功")
        
        # 测试训练相关导入
        from scene.scene import Scene
        from scene.net_vis import Visualizer
        from utils.config_utils import Config
        
        print("✓ 训练相关模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练脚本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("头部身体分离功能训练验证测试")
    print("=" * 60)
    
    # 运行所有测试
    test1_passed = test_training_functionality()
    test2_passed = test_dataset_loading()
    test3_passed = test_config_integration()
    test4_passed = test_training_script()
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"训练功能测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"数据集加载测试: {'✓ 通过' if test2_passed else '✗ 失败'}")
    print(f"配置集成测试: {'✓ 通过' if test3_passed else '✗ 失败'}")
    print(f"训练脚本测试: {'✓ 通过' if test4_passed else '✗ 失败'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("\n🎉 所有测试通过! 头部身体分离功能准备就绪")
        print("\n📋 训练命令示例:")
        print("python train.py --config config/SQ_02.yaml --data_dir /path/to/data --out_dir /path/to/output")
        exit(0)
    else:
        print("\n❌ 部分测试失败，请检查实现")
        exit(1)