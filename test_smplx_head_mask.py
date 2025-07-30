#!/usr/bin/env python3
"""
测试新的SMPL-X头部mask生成方法
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_smplx_head_mask_generation():
    """测试SMPL-X头部mask生成"""
    print("测试SMPL-X头部mask生成...")
    
    try:
        from utils.head_mask_utils import generate_head_mask, generate_geometric_head_mask, load_head_vertex_indices
        
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
        ]
        
        for height, width in test_sizes:
            print(f"\n测试图像尺寸: {height}x{width}")
            
            # 测试SMPL-X头部mask生成
            head_mask = generate_head_mask(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, height, width, device='cpu'
            )
            
            print(f"   SMPL-X mask形状: {head_mask.shape}")
            print(f"   头部像素数量: {head_mask.sum()}")
            print(f"   头部像素比例: {head_mask.sum() / (height * width) * 100:.1f}%")
            
            # 验证mask的基本属性
            assert head_mask.shape == (height, width), f"形状不匹配: {head_mask.shape} vs {(height, width)}"
            assert head_mask.dtype == bool, f"类型不匹配: {head_mask.dtype}"
            assert head_mask.sum() > 0, "头部mask不应该为空"
            
            # 测试几何fallback
            geo_mask = generate_geometric_head_mask(height, width)
            print(f"   几何mask像素数量: {geo_mask.sum()}")
            print(f"   几何mask像素比例: {geo_mask.sum() / (height * width) * 100:.1f}%")
            
            print("   ✓ 基本验证通过")
        
        # 测试头部顶点索引加载
        print(f"\n测试头部顶点索引加载...")
        head_vertex_indices = load_head_vertex_indices()
        print(f"   头部顶点数量: {len(head_vertex_indices)}")
        print(f"   顶点索引范围: {min(head_vertex_indices)} - {max(head_vertex_indices)}")
        
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
        # 验证head_mask_utils模块能被导入
        from utils.head_mask_utils import generate_head_mask, generate_geometric_head_mask
        print("✓ head_mask_utils模块导入成功")
        
        # 验证数据集模块更新
        from scene.dataset import AVRexDataset, ThumanDataset, ActorsHQDataset
        print("✓ 数据集模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集集成测试失败: {e}")
        return False

def test_head_mask_properties():
    """测试头部mask的属性"""
    print("\n测试头部mask属性...")
    
    try:
        from utils.head_mask_utils import generate_head_mask, generate_geometric_head_mask
        
        # 生成测试mask
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.zeros(3, dtype=np.float32)
        
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        height, width = 480, 640
        
        head_mask = generate_head_mask(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, height, width, device='cpu'
        )
        
        # 分析mask属性
        print(f"头部mask统计:")
        print(f"   形状: {head_mask.shape}")
        print(f"   数据类型: {head_mask.dtype}")
        print(f"   True像素数: {head_mask.sum()}")
        print(f"   False像素数: {(~head_mask).sum()}")
        print(f"   覆盖率: {head_mask.sum()/(height*width)*100:.2f}%")
        
        # 检查头部区域分布
        if head_mask.sum() > 0:
            y_indices, x_indices = np.where(head_mask)
            print(f"   Y范围: {y_indices.min()} - {y_indices.max()} (图像高度: {height})")
            print(f"   X范围: {x_indices.min()} - {x_indices.max()} (图像宽度: {width})")
            print(f"   中心Y: {y_indices.mean():.1f} (相对位置: {y_indices.mean()/height*100:.1f}%)")
            print(f"   中心X: {x_indices.mean():.1f} (相对位置: {x_indices.mean()/width*100:.1f}%)")
        
        # 验证头部区域应该在图像上部
        if head_mask.sum() > 0:
            y_center = np.where(head_mask)[0].mean()
            assert y_center < height * 0.6, f"头部中心应该在图像上部，但在 {y_center/height*100:.1f}%"
            print("   ✓ 头部位置验证通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 头部mask属性测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SMPL-X头部mask生成测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("SMPL-X头部mask生成", test_smplx_head_mask_generation),
        ("数据集集成", test_dataset_integration),
        ("头部mask属性", test_head_mask_properties),
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
        print("- 使用SMPL-X模型获取真实的头部顶点")
        print("- 通过头部顶点投影生成精确的头部掩膜")
        print("- 避免使用PyTorch3D渲染器，减少einsum错误")
        print("- 多进程环境下使用几何fallback")
        print("- 支持50维expression参数")
        print("\n现在可以重新训练，头部掩膜将基于真实的SMPL-X头部mesh:")
        print("python train.py --config ./config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_smplx_head_mask")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")