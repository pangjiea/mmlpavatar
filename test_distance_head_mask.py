#!/usr/bin/env python3
"""
测试基于距离的头部分离方法
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_distance_head_mask():
    """测试基于距离的头部mask生成"""
    print("测试基于距离的头部mask生成...")
    
    try:
        from utils.distance_head_mask import generate_head_mask_safe
        
        # 测试参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.zeros(3, dtype=np.float32)
        
        # 相机参数
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        
        # 测试不同的图像尺寸
        test_sizes = [
            (480, 640),   # VGA
            (720, 1280),  # HD
            (1080, 1920), # FullHD
        ]
        
        for height, width in test_sizes:
            print(f"\n测试图像尺寸: {height}x{width}")
            
            head_mask = generate_head_mask_safe(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, height, width, device='cpu'
            )
            
            print(f"   生成mask形状: {head_mask.shape}")
            print(f"   头部像素数量: {head_mask.sum()}")
            print(f"   头部像素比例: {head_mask.sum() / (height * width) * 100:.1f}%")
            
            # 验证mask的基本属性
            assert head_mask.shape == (height, width), f"形状不匹配: {head_mask.shape} vs {(height, width)}"
            assert head_mask.dtype == bool, f"类型不匹配: {head_mask.dtype}"
            assert head_mask.sum() > 0, "头部mask不应该为空"
            
            print("   ✓ 基本验证通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_integration():
    """测试数据集集成"""
    print("\n测试数据集集成...")
    
    try:
        # 这里我们不能真正测试数据集加载（没有数据），但可以验证导入
        from scene.dataset import AVRexDataset, ThumanDataset, ActorsHQDataset
        print("✓ 数据集模块导入成功")
        
        # 验证distance_head_mask模块能被导入
        from utils.distance_head_mask import generate_head_mask_safe
        print("✓ distance_head_mask模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集集成测试失败: {e}")
        return False

def test_head_body_separation_impact():
    """测试头部-身体分离对损失函数的影响"""
    print("\n测试头部-身体分离损失函数兼容性...")
    
    try:
        from utils.loss_utils import head_body_separation_loss
        print("✓ 头部-身体分离损失函数导入成功")
        
        # 模拟一个简单的GaussianModel（不完整，只是为了测试导入）
        class MockGaussianModel:
            def __init__(self):
                self.head_encoder_params = None
                self.body_encoder_params = None
                self.head_gs_mask = None
                self.body_gs_mask = None
        
        mock_gaussians = MockGaussianModel()
        
        # 测试损失函数不会崩溃
        loss, loss_dict = head_body_separation_loss(mock_gaussians)
        print(f"✓ 损失函数调用成功: {loss}, {list(loss_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ 头部-身体分离损失测试失败: {e}")
        return False

def test_head_mask_visualization():
    """测试头部mask的可视化效果"""
    print("\n测试头部mask可视化...")
    
    try:
        from utils.distance_head_mask import generate_head_mask_safe
        
        # 生成一个测试mask
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.zeros(3, dtype=np.float32)
        
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        height, width = 480, 640
        
        head_mask = generate_head_mask_safe(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, height, width, device='cpu'
        )
        
        # 简单的ASCII可视化（缩小版本）
        scale_factor = 8
        small_height = height // scale_factor
        small_width = width // scale_factor
        
        print(f"头部mask可视化 (缩放{scale_factor}x):")
        print("=" * (small_width + 2))
        
        for y in range(small_height):
            line = "|"
            for x in range(small_width):
                # 采样原始mask的对应区域
                orig_y = y * scale_factor
                orig_x = x * scale_factor
                sample_region = head_mask[orig_y:orig_y+scale_factor, orig_x:orig_x+scale_factor]
                
                if sample_region.sum() > (scale_factor * scale_factor) // 2:
                    line += "#"  # 头部区域
                else:
                    line += " "  # 非头部区域
            line += "|"
            print(line)
        
        print("=" * (small_width + 2))
        print("图例: # = 头部区域, 空格 = 非头部区域")
        
        return True
        
    except Exception as e:
        print(f"✗ 可视化测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("基于距离的头部分离方法测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("距离头部mask生成", test_distance_head_mask),
        ("数据集集成", test_dataset_integration),
        ("头部-身体分离损失兼容性", test_head_body_separation_impact),
        ("头部mask可视化", test_head_mask_visualization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过!")
        print("\n修复要点:")
        print("- 实现了基于距离的头部mask生成，避免SMPL-X渲染问题")
        print("- 头部区域使用椭圆形估计，位于图像上部中央")
        print("- 支持多种图像尺寸和设备模式")
        print("- 集成到所有数据集类中")
        print("- 与头部-身体分离损失函数兼容")
        print("\n现在可以重新训练，应该不会有einsum错误:")
        print("python train.py --config ./config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_distance_head_mask")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")