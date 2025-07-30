#!/usr/bin/env python3
"""
测试SQ_02数据集配置和头部身体分离功能的集成
"""

import torch
import numpy as np
import os
import sys
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_sq02_config():
    """测试SQ_02配置文件的头部身体分离参数"""
    print("开始测试SQ_02配置...")
    
    try:
        # 加载SQ_02配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        
        # 检查头部身体分离参数
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
                
        print("✓ SQ_02配置文件包含所有头部身体分离参数")
        return True
        
    except Exception as e:
        print(f"✗ SQ_02配置测试失败: {e}")
        return False

def test_dataset_compatibility():
    """测试数据集兼容性"""
    print("\n开始测试数据集兼容性...")
    
    try:
        from scene.gaussian_model import GaussianModel
        from scene.dataset import get_dataset_type
        from utils.loss_utils import head_body_separation_loss
        
        # 测试导入成功
        print("✓ 成功导入所有必要模块")
        
        # 测试数据集类型检测
        # 由于没有实际数据，我们只测试函数是否存在和可调用
        assert callable(get_dataset_type), "get_dataset_type应该是可调用的"
        print("✓ 数据集类型检测函数可用")
        
        # 测试GaussianModel的头部身体分离功能
        gaussians = GaussianModel()
        
        # 检查新增的属性是否存在
        required_attrs = [
            'head_encoder_params',
            'body_encoder_params', 
            'head_gs_mask',
            'body_gs_mask',
            'head_face_indices'
        ]
        
        for attr in required_attrs:
            assert hasattr(gaussians, attr), f"GaussianModel应该有属性 {attr}"
            print(f"✓ GaussianModel具有属性: {attr}")
            
        # 测试特征提取方法
        assert hasattr(gaussians, '_get_joint_features_for_type'), "应该有_get_joint_features_for_type方法"
        print("✓ 特征提取方法可用")
        
        # 测试损失函数
        assert callable(head_body_separation_loss), "head_body_separation_loss应该是可调用的"
        print("✓ 头部身体分离损失函数可用")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_simulation():
    """模拟训练流程测试"""
    print("\n开始模拟训练流程测试...")
    
    try:
        from scene.gaussian_model import GaussianModel
        from utils.loss_utils import head_body_separation_loss, head_consistency_loss
        from omegaconf import OmegaConf
        
        # 加载SQ_02配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        
        # 创建模拟的GaussianModel
        gaussians = GaussianModel()
        
        # 模拟训练数据
        gaussians.expression = torch.zeros(10, dtype=torch.float32)
        gaussians.jaw_pose = torch.zeros(3, dtype=torch.float32)
        gaussians.smpl_poses_cuda = torch.zeros(165, dtype=torch.float32, device='cuda')
        
        # 模拟head-body separation初始化（简化版）
        gaussians.head_encoder_params = None  # 开始时为None，正常情况
        gaussians.body_encoder_params = None
        gaussians.head_gs_mask = None
        gaussians.body_gs_mask = None
        
        print("✓ GaussianModel初始化成功")
        
        # 测试损失计算 - 应该返回0因为没有初始化head-body separation
        separation_loss, loss_dict = head_body_separation_loss(
            gaussians,
            alpha=args.lambda_head_body_feature_diff,
            beta=args.lambda_head_body_spatial_sep
        )
        
        print(f"✓ 头部身体分离损失计算: {separation_loss.item():.6f}")
        
        # 测试迭代条件
        iteration = 2000  # 大于iteration_head_body_loss
        if iteration > args.iteration_head_body_loss:
            print(f"✓ 迭代{iteration}大于开始条件{args.iteration_head_body_loss}，将应用头部身体分离损失")
        
        # 模拟头部掩膜
        head_mask_gt = torch.zeros(128, 128, device='cuda')
        consistency_loss = head_consistency_loss(gaussians, head_mask_gt, gamma=args.lambda_head_consistency)
        print(f"✓ 头部一致性损失计算: {consistency_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练流程模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_runtime_functionality():
    """测试运行时功能"""
    print("\n开始测试运行时功能...")
    
    try:
        # 测试头部掩膜生成功能
        from utils.head_mask_utils import generate_head_mask, get_neck_pose_from_full_pose
        
        print("✓ 头部掩膜工具导入成功")
        
        # 测试颈部姿态提取
        full_pose = torch.zeros(165, dtype=torch.float32)  # SMPL-X完整姿态
        neck_pose = get_neck_pose_from_full_pose(full_pose)
        
        assert neck_pose.shape == (3,), f"颈部姿态应该是3维，得到{neck_pose.shape}"
        print(f"✓ 颈部姿态提取成功: {neck_pose.shape}")
        
        # 测试身体部位到面部的映射
        import json
        body_parts_path = '/home/hello/code/mmlphuman/utils/smplx_body_parts_2_faces.json'
        with open(body_parts_path, 'r') as f:
            body_parts = json.load(f)
            
        assert 'head' in body_parts, "应该包含head部位"
        head_faces = body_parts['head']
        print(f"✓ 头部面部索引加载成功: {len(head_faces)}个面")
        
        return True
        
    except Exception as e:
        print(f"✗ 运行时功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SQ_02数据集头部身体分离集成测试")
    print("=" * 60)
    
    # 运行所有测试
    test1_passed = test_sq02_config()
    test2_passed = test_dataset_compatibility()
    test3_passed = test_training_simulation()
    test4_passed = test_runtime_functionality()
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"SQ_02配置测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"数据集兼容性测试: {'✓ 通过' if test2_passed else '✗ 失败'}")
    print(f"训练流程模拟测试: {'✓ 通过' if test3_passed else '✗ 失败'}")
    print(f"运行时功能测试: {'✓ 通过' if test4_passed else '✗ 失败'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("\n🎉 所有测试通过! SQ_02数据集已支持头部身体分离功能")
        exit(0)
    else:
        print("\n❌ 部分测试失败，请检查实现")
        exit(1)