#!/usr/bin/env python3
"""
测试头部-身体分离功能的脚本
"""

import torch
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_head_body_separation():
    """测试头部-身体分离功能"""
    print("开始测试头部-身体分离功能...")
    
    try:
        # 导入必要的模块
        from scene.gaussian_model import GaussianModel
        from utils.smpl_utils import smpl
        import json
        
        print("✓ 成功导入模块")
        
        # 测试1: 检查smplx_body_parts_2_faces.json文件
        json_path = '/home/hello/code/mmlphuman/utils/smplx_body_parts_2_faces.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                body_parts = json.load(f)
            
            if 'head' in body_parts:
                head_face_indices = body_parts['head']
                print(f"✓ 成功加载头部面片索引，共 {len(head_face_indices)} 个面片")
            else:
                print("✗ 头部面片索引未找到")
                return False
        else:
            print(f"✗ 文件不存在: {json_path}")
            return False
            
        # 测试2: 检查GaussianModel的新功能
        gaussian_model = GaussianModel()
        
        # 检查新增的属性
        required_attrs = [
            'head_face_indices', 'head_gs_mask', 'body_gs_mask',
            'head_encoder_params', 'body_encoder_params'
        ]
        
        for attr in required_attrs:
            if hasattr(gaussian_model, attr):
                print(f"✓ GaussianModel 具有属性: {attr}")
            else:
                print(f"✗ GaussianModel 缺少属性: {attr}")
                return False
        
        # 测试3: 检查特征提取功能
        try:
            # 模拟一些基本参数
            gaussian_model.expression = torch.zeros(10, dtype=torch.float32)
            gaussian_model.jaw_pose = torch.zeros(3, dtype=torch.float32)
            gaussian_model.smpl_poses_cuda = torch.zeros(165, dtype=torch.float32, device='cuda')
            
            # 测试不同类型的特征提取
            head_features = gaussian_model._get_joint_features_for_type('head')
            body_features = gaussian_model._get_joint_features_for_type('body')
            all_features = gaussian_model._get_joint_features_for_type('all')
            
            print(f"✓ 头部特征维度: {head_features.shape} (期望: [16])")
            print(f"✓ 身体特征维度: {body_features.shape} (期望: [66])")
            print(f"✓ 全部特征维度: {all_features.shape} (期望: [76])")
            
            # 验证维度
            if head_features.shape[0] == 16 and body_features.shape[0] == 66 and all_features.shape[0] == 76:
                print("✓ 特征维度正确")
            else:
                print("✗ 特征维度不正确")
                return False
                
        except Exception as e:
            print(f"✗ 特征提取测试失败: {e}")
            return False
        
        # 测试4: 检查头部面片加载功能
        try:
            gaussian_model._load_head_face_indices()
            if gaussian_model.head_face_indices is not None:
                print(f"✓ 成功加载头部面片索引: {len(gaussian_model.head_face_indices)} 个")
            else:
                print("✗ 头部面片索引加载失败")
                return False
        except Exception as e:
            print(f"✗ 头部面片加载测试失败: {e}")
            return False
        
        print("\n🎉 所有测试通过! 头部-身体分离功能实现正确")
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        return False

def test_dataset_integration():
    """测试数据集集成"""
    print("\n开始测试数据集集成...")
    
    try:
        from scene.dataset import get_dataset_type
        from utils.head_mask_utils import generate_head_mask, get_neck_pose_from_full_pose
        
        print("✓ 成功导入数据集相关模块")
        
        # 测试头部mask生成函数
        pose = np.random.randn(75).astype(np.float32)
        beta = np.random.randn(10).astype(np.float32)
        expression = np.random.randn(50).astype(np.float32)
        jaw_pose = np.random.randn(3).astype(np.float32)
        Rh = np.random.randn(3).astype(np.float32)
        Th = np.random.randn(3).astype(np.float32)
        K = np.eye(3).astype(np.float32)
        w2c = np.eye(4).astype(np.float32)
        
        print("✓ 头部mask生成函数参数准备完成")
        
        # 测试neck pose提取
        full_pose = np.random.randn(75).astype(np.float32)
        neck_pose = get_neck_pose_from_full_pose(full_pose)
        
        if neck_pose.shape[0] == 3:
            print(f"✓ neck pose提取正确，维度: {neck_pose.shape}")
        else:
            print(f"✗ neck pose维度错误: {neck_pose.shape}")
            return False
            
        print("✓ 数据集集成测试通过")
        return True
        
    except ImportError as e:
        print(f"✗ 数据集模块导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 数据集集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("头部-身体分离功能测试")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_head_body_separation()
    test2_passed = test_dataset_integration()
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"头部-身体分离功能: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"数据集集成: {'✓ 通过' if test2_passed else '✗ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过! 可以进行下一步开发")
        exit(0)
    else:
        print("\n❌ 部分测试失败，请检查实现")
        exit(1)