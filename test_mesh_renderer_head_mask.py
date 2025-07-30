#!/usr/bin/env python3
"""
测试新的mesh渲染器头部掩膜生成功能
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_mesh_renderer_head_mask():
    """测试mesh渲染器头部掩膜生成"""
    print("🔍 开始测试mesh渲染器头部掩膜生成...")
    
    try:
        # 导入头部掩膜生成函数
        from utils.head_mask_utils import (
            generate_head_mask_with_mesh_renderer,
            generate_head_mask,
            render_head_mesh_to_mask,
            generate_geometric_head_mask
        )
        print("✓ 成功导入头部掩膜生成函数")
        
        # 创建测试参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)  # 测试50维expression
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.array([0, 0, 2], dtype=np.float32)  # 稍微远离相机
        
        # 相机参数
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        image_height, image_width = 480, 640
        
        print("✓ 创建测试参数完成")
        
        # 测试1: 高级mesh渲染器（带Z-buffer）
        print("\n🧪 测试1: 高级mesh渲染器（带Z-buffer）")
        try:
            head_mask_advanced = generate_head_mask_with_mesh_renderer(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, image_height, image_width, 
                device='cpu', use_advanced_renderer=True, use_z_buffer=True
            )
            
            if head_mask_advanced is not None:
                print(f"✓ 高级渲染器成功: {head_mask_advanced.sum()} 像素")
                save_mask_visualization(head_mask_advanced, "advanced_zbuffer_head_mask.png")
            else:
                print("❌ 高级渲染器返回None")
                
        except Exception as e:
            print(f"❌ 高级渲染器测试失败: {e}")
        
        # 测试2: 高级mesh渲染器（无Z-buffer）
        print("\n🧪 测试2: 高级mesh渲染器（无Z-buffer）")
        try:
            head_mask_simple_advanced = generate_head_mask_with_mesh_renderer(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, image_height, image_width, 
                device='cpu', use_advanced_renderer=True, use_z_buffer=False
            )
            
            if head_mask_simple_advanced is not None:
                print(f"✓ 简化高级渲染器成功: {head_mask_simple_advanced.sum()} 像素")
                save_mask_visualization(head_mask_simple_advanced, "advanced_simple_head_mask.png")
            else:
                print("❌ 简化高级渲染器返回None")
                
        except Exception as e:
            print(f"❌ 简化高级渲染器测试失败: {e}")
        
        # 测试3: 简单TalkBody4D风格渲染器
        print("\n🧪 测试3: 简单TalkBody4D风格渲染器")
        try:
            head_mask_talkbody = generate_head_mask_with_mesh_renderer(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, image_height, image_width, 
                device='cpu', use_advanced_renderer=False, use_z_buffer=False
            )
            
            if head_mask_talkbody is not None:
                print(f"✓ TalkBody4D风格渲染器成功: {head_mask_talkbody.sum()} 像素")
                save_mask_visualization(head_mask_talkbody, "talkbody_head_mask.png")
            else:
                print("❌ TalkBody4D风格渲染器返回None")
                
        except Exception as e:
            print(f"❌ TalkBody4D风格渲染器测试失败: {e}")
        
        # 测试4: 向后兼容的原始函数
        print("\n🧪 测试4: 向后兼容的原始函数")
        try:
            head_mask_original = generate_head_mask(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, image_height, image_width, device='cpu'
            )
            
            if head_mask_original is not None:
                print(f"✓ 原始函数成功: {head_mask_original.sum()} 像素")
                save_mask_visualization(head_mask_original, "original_head_mask.png")
            else:
                print("❌ 原始函数返回None")
                
        except Exception as e:
            print(f"❌ 原始函数测试失败: {e}")
        
        # 测试5: 几何fallback
        print("\n🧪 测试5: 几何fallback")
        try:
            head_mask_geometric = generate_geometric_head_mask(image_height, image_width)
            print(f"✓ 几何fallback成功: {head_mask_geometric.sum()} 像素")
            save_mask_visualization(head_mask_geometric, "geometric_head_mask.png")
            
        except Exception as e:
            print(f"❌ 几何fallback测试失败: {e}")
        
        print("\n✅ mesh渲染器头部掩膜生成测试完成!")
        print("📁 可视化结果保存在当前目录")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def save_mask_visualization(mask, filename):
    """保存掩膜可视化"""
    try:
        if mask is None:
            return
            
        # 创建可视化
        plt.figure(figsize=(10, 8))
        plt.imshow(mask, cmap='gray')
        plt.title(f'Head Mask: {mask.sum()} pixels')
        plt.colorbar()
        plt.axis('off')
        
        # 保存图像
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存可视化: {filename}")
        
    except Exception as e:
        print(f"❌ 保存可视化失败: {e}")

def test_mesh_renderer_components():
    """测试mesh渲染器的各个组件"""
    print("\n🔧 测试mesh渲染器组件...")
    
    try:
        from utils.head_mask_utils import (
            point_in_triangle,
            compute_barycentric_coordinates,
            render_triangle_with_zbuffer
        )
        
        # 测试点在三角形内判断
        triangle = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
        
        # 测试点
        test_points = [
            (5, 3),   # 应该在内部
            (15, 5),  # 应该在外部
            (2, 1),   # 应该在内部
        ]
        
        print("🔍 测试点在三角形内判断:")
        for i, (x, y) in enumerate(test_points):
            inside = point_in_triangle(x, y, triangle)
            print(f"  点 ({x}, {y}): {'内部' if inside else '外部'}")
        
        # 测试重心坐标计算
        print("\n🔍 测试重心坐标计算:")
        for i, (x, y) in enumerate(test_points[:2]):  # 只测试前两个点
            if point_in_triangle(x, y, triangle):
                barycentric = compute_barycentric_coordinates(x, y, triangle)
                print(f"  点 ({x}, {y}) 重心坐标: {barycentric}")
                print(f"    坐标和: {barycentric.sum():.3f}")
        
        print("✅ mesh渲染器组件测试完成!")
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 开始mesh渲染器头部掩膜测试")
    test_mesh_renderer_head_mask()
    test_mesh_renderer_components()
    print("🎉 所有测试完成!")
