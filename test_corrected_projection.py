#!/usr/bin/env python3
"""
测试修正后的TalkBody4D投影方法
验证头部掩膜投影位置是否正确
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_corrected_projection():
    """测试修正后的投影方法"""
    print("🔍 开始测试修正后的TalkBody4D投影...")
    
    try:
        from utils.head_mask_utils import (
            generate_head_mask_with_mesh_renderer,
            extract_camera_params_from_w2c,
            talkbody4d_projection_exact
        )
        
        # 创建测试参数 - 使用更真实的参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        
        # 设置一个合理的头部姿态
        Rh = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 正面朝向
        Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # 距离相机2米
        
        # 相机参数 - 模拟SQ02数据集的相机
        K = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 构造w2c矩阵 - 模拟数据集的格式
        # 相机在原点，朝向+Z方向
        R_cam = np.eye(3, dtype=np.float32)
        T_cam = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_cam
        w2c[:3, 3] = T_cam
        
        image_height, image_width = 480, 640
        
        print("✓ 创建测试参数完成")
        print(f"  Rh: {Rh}")
        print(f"  Th: {Th}")
        print(f"  相机内参 K:\n{K}")
        print(f"  w2c矩阵:\n{w2c}")
        
        # 测试相机参数提取
        print("\n🔧 测试相机参数提取...")
        R_extracted, T_extracted = extract_camera_params_from_w2c(w2c)
        
        # 测试1: 使用修正后的简单渲染器
        print("\n🧪 测试1: 修正后的简单渲染器")
        try:
            head_mask_simple = generate_head_mask_with_mesh_renderer(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, image_height, image_width,
                device='cpu',
                use_advanced_renderer=False,  # 使用简单渲染器
                use_z_buffer=False
            )
            
            if head_mask_simple is not None:
                pixel_count = head_mask_simple.sum()
                percentage = (pixel_count / (image_height * image_width)) * 100
                print(f"✓ 简单渲染器成功: {pixel_count} 像素 ({percentage:.1f}%)")
                save_mask_with_info(head_mask_simple, "corrected_simple_head_mask.png", 
                                  f"Simple Renderer: {pixel_count} pixels")
            else:
                print("❌ 简单渲染器返回None")
                
        except Exception as e:
            print(f"❌ 简单渲染器测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试2: 使用修正后的高级渲染器
        print("\n🧪 测试2: 修正后的高级渲染器")
        try:
            head_mask_advanced = generate_head_mask_with_mesh_renderer(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, image_height, image_width,
                device='cpu',
                use_advanced_renderer=True,  # 使用高级渲染器
                use_z_buffer=True
            )
            
            if head_mask_advanced is not None:
                pixel_count = head_mask_advanced.sum()
                percentage = (pixel_count / (image_height * image_width)) * 100
                print(f"✓ 高级渲染器成功: {pixel_count} 像素 ({percentage:.1f}%)")
                save_mask_with_info(head_mask_advanced, "corrected_advanced_head_mask.png", 
                                  f"Advanced Renderer: {pixel_count} pixels")
            else:
                print("❌ 高级渲染器返回None")
                
        except Exception as e:
            print(f"❌ 高级渲染器测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试3: 直接测试投影函数
        print("\n🧪 测试3: 直接测试TalkBody4D投影")
        try:
            # 创建一些测试顶点
            test_vertices = np.array([
                [0.0, 0.0, 2.0],    # 正前方
                [0.1, 0.1, 2.0],    # 稍微偏移
                [-0.1, 0.1, 2.0],   # 另一个偏移
                [0.0, 0.2, 2.0],    # 头顶
                [0.0, -0.1, 2.0],   # 下巴
            ], dtype=np.float32)
            
            pixels, valid_mask = talkbody4d_projection_exact(
                test_vertices, K, R_extracted, T_extracted, image_width, image_height
            )
            
            if pixels is not None:
                print(f"✓ 投影测试成功")
                print(f"  测试顶点投影结果:")
                for i, (vertex, pixel, valid) in enumerate(zip(test_vertices, pixels, valid_mask)):
                    print(f"    顶点{i}: {vertex} -> 像素{pixel} (有效: {valid})")
            else:
                print("❌ 投影测试失败")
                
        except Exception as e:
            print(f"❌ 投影测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ 修正后的投影测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def save_mask_with_info(mask, filename, title):
    """保存掩膜并添加信息"""
    try:
        if mask is None:
            return
            
        plt.figure(figsize=(10, 8))
        plt.imshow(mask, cmap='gray')
        plt.title(title, fontsize=14, pad=20)
        plt.colorbar()
        plt.axis('off')
        
        # 添加统计信息
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = mask.sum()
        percentage = (mask_pixels / total_pixels) * 100
        
        plt.text(10, 30, f'Total: {mask_pixels:,} pixels', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, color='black')
        plt.text(10, 60, f'Coverage: {percentage:.1f}%', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, color='black')
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存可视化: {filename}")
        
    except Exception as e:
        print(f"❌ 保存可视化失败: {e}")

def test_different_poses():
    """测试不同姿态下的投影"""
    print("\n🎭 测试不同姿态下的投影...")
    
    try:
        from utils.head_mask_utils import generate_head_mask_with_mesh_renderer
        
        # 基础参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        
        K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        image_height, image_width = 480, 640
        
        # 测试不同的头部姿态
        test_poses = [
            {"name": "正面", "Rh": [0.0, 0.0, 0.0], "Th": [0.0, 0.0, 2.0]},
            {"name": "左转", "Rh": [0.0, 0.3, 0.0], "Th": [0.0, 0.0, 2.0]},
            {"name": "右转", "Rh": [0.0, -0.3, 0.0], "Th": [0.0, 0.0, 2.0]},
            {"name": "抬头", "Rh": [0.2, 0.0, 0.0], "Th": [0.0, 0.0, 2.0]},
            {"name": "低头", "Rh": [-0.2, 0.0, 0.0], "Th": [0.0, 0.0, 2.0]},
        ]
        
        results = []
        
        for test_pose in test_poses:
            print(f"\n🎯 测试姿态: {test_pose['name']}")
            
            Rh = np.array(test_pose['Rh'], dtype=np.float32)
            Th = np.array(test_pose['Th'], dtype=np.float32)
            
            try:
                head_mask = generate_head_mask_with_mesh_renderer(
                    pose, beta, expression, jaw_pose, Rh, Th,
                    K, w2c, image_height, image_width,
                    device='cpu', use_advanced_renderer=False, use_z_buffer=False
                )
                
                if head_mask is not None:
                    pixel_count = head_mask.sum()
                    percentage = (pixel_count / (image_height * image_width)) * 100
                    print(f"✓ {test_pose['name']}: {pixel_count} 像素 ({percentage:.1f}%)")
                    
                    filename = f"pose_{test_pose['name']}_head_mask.png"
                    save_mask_with_info(head_mask, filename, f"{test_pose['name']} - {pixel_count} pixels")
                    
                    results.append({
                        'name': test_pose['name'],
                        'pixels': pixel_count,
                        'percentage': percentage
                    })
                else:
                    print(f"❌ {test_pose['name']}: 生成失败")
                    
            except Exception as e:
                print(f"❌ {test_pose['name']} 测试失败: {e}")
        
        # 打印总结
        print(f"\n📊 姿态测试总结:")
        for result in results:
            print(f"  {result['name']}: {result['pixels']} 像素 ({result['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"❌ 姿态测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 开始修正后的投影测试")
    test_corrected_projection()
    test_different_poses()
    print("🎉 测试完成!")
