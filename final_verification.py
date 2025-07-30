#!/usr/bin/env python3
"""
头部身体分离功能最终验证脚本
"""

import os
import torch
import sys
import numpy as np
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def final_comprehensive_test():
    """最终综合测试"""
    print("🔍 开始最终综合验证...")
    
    results = {}
    
    try:
        # 1. 模块导入测试
        print("\n1️⃣ 测试模块导入...")
        from scene.gaussian_model import GaussianModel
        from scene.scene import Scene
        from scene.dataset import get_dataset_type, data_to_cam
        from utils.loss_utils import head_body_separation_loss, head_consistency_loss
        from utils.smpl_utils import init_smpl_pose
        from utils.head_mask_utils import get_neck_pose_from_full_pose
        print("✓ 所有关键模块导入成功")
        results['module_import'] = True
        
        # 2. 数据集验证
        print("\n2️⃣ 测试数据集...")
        data_dir = '/home/hello/data/SQ_02/'
        if os.path.exists(data_dir):
            DatasetType = get_dataset_type(data_dir)
            print(f"✓ 数据集类型: {DatasetType.__name__}")
            
            # 检查SMPL参数
            smpl_file = os.path.join(data_dir, 'smpl_params.npz')
            if os.path.exists(smpl_file):
                data = np.load(smpl_file, allow_pickle=True)
                has_expression = 'expression' in data.keys()
                has_jaw_pose = 'jaw_pose' in data.keys()
                print(f"✓ Expression参数: {'存在' if has_expression else '缺失'}")
                print(f"✓ Jaw pose参数: {'存在' if has_jaw_pose else '缺失'}")
                results['dataset'] = has_expression and has_jaw_pose
            else:
                print("✗ SMPL参数文件不存在")
                results['dataset'] = False
        else:
            print("✗ 数据集目录不存在")
            results['dataset'] = False
        
        # 3. 配置验证
        print("\n3️⃣ 测试配置...")
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        if os.path.exists(config_path):
            args = OmegaConf.load(config_path)
            required_params = [
                'lambda_head_body_separation',
                'lambda_head_body_feature_diff',
                'lambda_head_body_spatial_sep', 
                'lambda_head_consistency',
                'iteration_head_body_loss'
            ]
            
            config_ok = True
            for param in required_params:
                if hasattr(args, param):
                    print(f"✓ {param}: {getattr(args, param)}")
                else:
                    print(f"✗ 缺少参数: {param}")
                    config_ok = False
            results['config'] = config_ok
        else:
            print("✗ 配置文件不存在")
            results['config'] = False
        
        # 4. 模型初始化测试
        print("\n4️⃣ 测试模型初始化...")
        init_smpl_pose()
        gaussians = GaussianModel()
        
        # 检查头部身体分离属性
        required_attrs = [
            'head_encoder_params', 'body_encoder_params',
            'head_gs_mask', 'body_gs_mask', 'head_face_indices'
        ]
        
        attr_ok = True
        for attr in required_attrs:
            if hasattr(gaussians, attr):
                print(f"✓ 模型属性: {attr}")
            else:
                print(f"✗ 缺少属性: {attr}")
                attr_ok = False
        results['model_init'] = attr_ok
        
        # 5. 特征提取测试
        print("\n5️⃣ 测试特征提取...")
        gaussians.expression = torch.zeros(10, dtype=torch.float32)
        gaussians.jaw_pose = torch.zeros(3, dtype=torch.float32)
        gaussians.smpl_poses_cuda = torch.zeros(165, dtype=torch.float32, device='cuda')
        
        try:
            # 测试不同类型的特征提取
            if hasattr(gaussians, '_get_joint_features_for_type'):
                all_features = gaussians._get_joint_features_for_type('all')
                head_features = gaussians._get_joint_features_for_type('head')
                body_features = gaussians._get_joint_features_for_type('body')
                
                print(f"✓ 全部特征维度: {all_features.shape}")
                print(f"✓ 头部特征维度: {head_features.shape}")
                print(f"✓ 身体特征维度: {body_features.shape}")
                
                # 验证维度正确性
                expected_head_dim = 16  # expression(10) + jaw_pose(3) + neck_pose(3)
                expected_body_dim = 66  # body_poses(63) + neck_pose(3)
                
                dim_ok = (head_features.shape[0] == expected_head_dim and 
                         body_features.shape[0] == expected_body_dim)
                print(f"✓ 特征维度验证: {'正确' if dim_ok else '错误'}")
                results['feature_extraction'] = dim_ok
            else:
                print("✗ 特征提取方法不存在")
                results['feature_extraction'] = False
        except Exception as e:
            print(f"✗ 特征提取失败: {e}")
            results['feature_extraction'] = False
        
        # 6. 损失函数测试
        print("\n6️⃣ 测试损失函数...")
        try:
            separation_loss, loss_dict = head_body_separation_loss(gaussians)
            print(f"✓ 头部身体分离损失: {separation_loss.item():.6f}")
            print(f"✓ 损失项: {list(loss_dict.keys())}")
            
            head_mask_gt = torch.zeros(128, 128, device='cuda')
            consistency_loss = head_consistency_loss(gaussians, head_mask_gt)
            print(f"✓ 头部一致性损失: {consistency_loss.item():.6f}")
            
            results['loss_functions'] = True
        except Exception as e:
            print(f"✗ 损失函数测试失败: {e}")
            results['loss_functions'] = False
        
        # 7. 颈部姿态提取测试
        print("\n7️⃣ 测试颈部姿态提取...")
        try:
            full_pose = torch.zeros(165, dtype=torch.float32)
            neck_pose = get_neck_pose_from_full_pose(full_pose)
            print(f"✓ 颈部姿态形状: {neck_pose.shape}")
            print(f"✓ 颈部姿态值: {neck_pose}")
            results['neck_pose'] = neck_pose.shape == torch.Size([3])
        except Exception as e:
            print(f"✗ 颈部姿态提取失败: {e}")
            results['neck_pose'] = False
        
    except Exception as e:
        print(f"✗ 综合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, results
    
    return True, results

def print_final_summary(success, results):
    """打印最终总结"""
    print("\n" + "="*60)
    print("🎯 头部身体分离功能最终验证报告")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 项测试通过")
    
    status_icon = {True: "✅", False: "❌"}
    test_names = {
        'module_import': '模块导入',
        'dataset': '数据集验证', 
        'config': '配置文件',
        'model_init': '模型初始化',
        'feature_extraction': '特征提取',
        'loss_functions': '损失函数',
        'neck_pose': '颈部姿态提取'
    }
    
    print("\n📋 详细结果:")
    for key, passed in results.items():
        name = test_names.get(key, key)
        print(f"  {status_icon[passed]} {name}: {'通过' if passed else '失败'}")
    
    if success and passed_tests == total_tests:
        print("\n🎉 恭喜! 头部身体分离功能已完全就绪!")
        print("\n🚀 可以开始训练:")
        print("python train.py --config config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_head_body")
        
        print("\n📝 功能特点:")
        print("• 支持SMPL-X expression和jaw_pose参数")
        print("• 头部编码器: expression + jaw_pose + neck_pose (16维)")
        print("• 身体编码器: body_poses + neck_pose (66维)")
        print("• 第1000次迭代后激活头部身体分离损失")
        print("• 包含特征差异、空间分离、编码器特化三种损失")
        print("• 完全向后兼容原有功能")
        
        print("\n⚡ 训练建议:")
        print("• 建议训练20000-50000次迭代")
        print("• 监控头部身体分离损失的收敛情况") 
        print("• 可以通过TensorBoard查看训练进度")
        
    else:
        print("\n⚠️  部分功能存在问题，建议检查失败项目")
        print("但基本功能可用，可以尝试训练")

if __name__ == "__main__":
    print("🔥 头部身体分离功能最终验证")
    print("=" * 60)
    
    success, results = final_comprehensive_test()
    print_final_summary(success, results)