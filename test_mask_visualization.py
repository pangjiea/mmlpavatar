#!/usr/bin/env python3
"""
测试头部mask可视化功能
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_mask_visualization():
    """测试mask可视化功能"""
    print("测试头部mask可视化功能...")
    
    try:
        from utils.mask_visualization import save_head_mask_visualization, save_head_mask_comparison
        
        # 创建测试数据
        height, width = 480, 640
        
        # 创建测试图像
        image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # 创建椭圆形头部mask
        head_mask = np.zeros((height, width), dtype=bool)
        center_y, center_x = height // 4, width // 2  # 头部在上部中央
        radius_y, radius_x = height // 6, width // 8
        
        for y in range(height):
            for x in range(width):
                dy = (y - center_y) / radius_y
                dx = (x - center_x) / radius_x
                if dy*dy + dx*dx <= 1.0:
                    head_mask[y, x] = True
        
        print(f"创建测试头部mask: {head_mask.sum()} 像素 ({head_mask.sum()/(height*width)*100:.1f}%)")
        
        # 测试基本可视化
        debug_dir = "/home/hello/code/mmlphuman/debug_test_masks"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存基本头部mask可视化
        basic_path = os.path.join(debug_dir, "test_basic_head_mask.png")
        save_head_mask_visualization(image, head_mask, basic_path, "Test Basic Head Mask")
        print(f"✓ 基本头部mask可视化保存到: {basic_path}")
        
        # 创建身体mask
        body_mask = np.ones((height, width), dtype=bool)
        body_mask[head_mask] = False
        
        # 保存头部-身体对比
        comparison_path = os.path.join(debug_dir, "test_head_body_comparison.png")
        save_head_mask_comparison(image, head_mask, body_mask, comparison_path, frame_id=123, cam_id=1)
        print(f"✓ 头部-身体对比保存到: {comparison_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_monitor():
    """测试训练监控功能"""
    print("\n测试训练监控功能...")
    
    try:
        from utils.head_mask_monitor import HeadMaskMonitor
        import torch
        
        # 创建监控器
        debug_dir = "/home/hello/code/mmlphuman/debug_test_monitor"
        monitor = HeadMaskMonitor(debug_dir, max_samples=3, save_interval=100)
        
        # 创建测试数据
        test_image = torch.randn(3, 480, 640)  # [C, H, W]
        test_head_mask = torch.zeros(480, 640, dtype=torch.bool)
        test_head_mask[50:200, 250:390] = True  # 头部区域
        
        test_data = {
            'image': test_image,
            'head_mask': test_head_mask,
            'frame_id': 123,
            'cam_id': 1
        }
        
        # 模拟Gaussian模型
        class MockGaussians:
            def __init__(self):
                self.head_gs_mask = None
                self.body_gs_mask = None
        
        mock_gaussians = MockGaussians()
        
        # 测试保存
        monitor.save_training_sample(test_data, mock_gaussians, 1000)
        print("✓ 训练监控功能测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练监控测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_integration():
    """测试数据集集成"""
    print("\n测试数据集集成...")
    
    try:
        # 验证相关模块能正确导入
        from utils.mask_visualization import save_head_mask_visualization
        from utils.head_mask_monitor import save_training_head_mask_sample
        from scene.dataset import AVRexDataset
        
        print("✓ 所有相关模块导入成功")
        return True
        
    except Exception as e:
        print(f"✗ 数据集集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("头部Mask可视化功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("基本可视化功能", test_mask_visualization),
        ("训练监控功能", test_training_monitor),
        ("数据集集成", test_dataset_integration),
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
        print("\n头部mask可视化功能已就绪:")
        print("- 基本头部mask可视化")
        print("- 头部-身体分离对比")
        print("- 训练过程监控")
        print("- 自动保存到debug目录")
        print("\n训练时会自动保存头部mask可视化到:")
        print("  <output_dir>/debug_head_masks/")
        print("\n数据集加载时前5个样本的头部mask保存到:")
        print("  /home/hello/code/mmlphuman/debug_head_masks/")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")