#!/usr/bin/env python3
"""
演示新的mesh渲染器头部掩膜生成功能
展示不同渲染模式的效果对比
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def demo_mesh_renderer():
    """演示mesh渲染器的不同模式"""
    print("🎬 开始mesh渲染器演示...")
    
    try:
        from utils.head_mask_utils import (
            generate_head_mask_with_mesh_renderer,
            generate_geometric_head_mask
        )
        
        # 创建测试参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.random.normal(0, 0.1, 10).astype(np.float32)  # 随机形状参数
        expression = np.random.normal(0, 0.05, 50).astype(np.float32)  # 随机表情
        jaw_pose = np.array([0.1, 0, 0], dtype=np.float32)  # 轻微张嘴
        Rh = np.array([0.1, 0.05, 0], dtype=np.float32)  # 轻微转头
        Th = np.array([0, 0, 2.5], dtype=np.float32)  # 距离相机2.5米
        
        # 相机参数
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        image_height, image_width = 480, 640
        
        print("✓ 创建测试参数完成")
        
        # 测试不同的渲染模式
        render_modes = [
            {
                'name': '高级渲染器 + Z-buffer',
                'use_advanced_renderer': True,
                'use_z_buffer': True,
                'color': 'red'
            },
            {
                'name': '高级渲染器（无Z-buffer）',
                'use_advanced_renderer': True,
                'use_z_buffer': False,
                'color': 'blue'
            },
            {
                'name': 'TalkBody4D风格',
                'use_advanced_renderer': False,
                'use_z_buffer': False,
                'color': 'green'
            }
        ]
        
        results = {}
        
        # 生成不同模式的掩膜
        for mode in render_modes:
            print(f"\n🎨 测试: {mode['name']}")
            try:
                head_mask = generate_head_mask_with_mesh_renderer(
                    pose, beta, expression, jaw_pose, Rh, Th,
                    K, w2c, image_height, image_width,
                    device='cpu',
                    use_advanced_renderer=mode['use_advanced_renderer'],
                    use_z_buffer=mode['use_z_buffer']
                )
                
                if head_mask is not None:
                    pixel_count = head_mask.sum()
                    percentage = (pixel_count / (image_height * image_width)) * 100
                    print(f"✓ {mode['name']}: {pixel_count} 像素 ({percentage:.1f}%)")
                    results[mode['name']] = {
                        'mask': head_mask,
                        'pixels': pixel_count,
                        'percentage': percentage,
                        'color': mode['color']
                    }
                else:
                    print(f"❌ {mode['name']}: 生成失败")
                    
            except Exception as e:
                print(f"❌ {mode['name']} 测试失败: {e}")
        
        # 添加几何fallback作为对比
        print(f"\n🎨 测试: 几何Fallback")
        try:
            geometric_mask = generate_geometric_head_mask(image_height, image_width)
            pixel_count = geometric_mask.sum()
            percentage = (pixel_count / (image_height * image_width)) * 100
            print(f"✓ 几何Fallback: {pixel_count} 像素 ({percentage:.1f}%)")
            results['几何Fallback'] = {
                'mask': geometric_mask,
                'pixels': pixel_count,
                'percentage': percentage,
                'color': 'orange'
            }
        except Exception as e:
            print(f"❌ 几何Fallback测试失败: {e}")
        
        # 创建对比可视化
        create_comparison_visualization(results, image_height, image_width)
        
        # 创建统计图表
        create_statistics_chart(results)
        
        print("\n✅ mesh渲染器演示完成!")
        print("📁 可视化结果保存为:")
        print("  - mesh_renderer_comparison.png (掩膜对比)")
        print("  - mesh_renderer_statistics.png (统计图表)")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def create_comparison_visualization(results, height, width):
    """创建不同渲染模式的对比可视化"""
    try:
        n_results = len(results)
        if n_results == 0:
            return
        
        # 计算子图布局
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_results == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制每个结果
        for i, (name, data) in enumerate(results.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            mask = data['mask']
            pixels = data['pixels']
            percentage = data['percentage']
            
            # 显示掩膜
            im = ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{name}\n{pixels} 像素 ({percentage:.1f}%)', 
                        fontsize=12, pad=10)
            ax.axis('off')
            
            # 添加颜色边框
            rect = Rectangle((0, 0), width-1, height-1, 
                           linewidth=3, edgecolor=data['color'], 
                           facecolor='none')
            ax.add_patch(rect)
        
        # 隐藏多余的子图
        for i in range(n_results, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('mesh_renderer_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ 保存对比可视化: mesh_renderer_comparison.png")
        
    except Exception as e:
        print(f"❌ 创建对比可视化失败: {e}")

def create_statistics_chart(results):
    """创建统计图表"""
    try:
        if len(results) == 0:
            return
        
        # 提取数据
        names = list(results.keys())
        pixels = [results[name]['pixels'] for name in names]
        percentages = [results[name]['percentage'] for name in names]
        colors = [results[name]['color'] for name in names]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 像素数量柱状图
        bars1 = ax1.bar(names, pixels, color=colors, alpha=0.7)
        ax1.set_title('头部掩膜像素数量对比', fontsize=14, pad=20)
        ax1.set_ylabel('像素数量', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, pixel in zip(bars1, pixels):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{pixel:,}', ha='center', va='bottom', fontsize=10)
        
        # 百分比柱状图
        bars2 = ax2.bar(names, percentages, color=colors, alpha=0.7)
        ax2.set_title('头部掩膜覆盖率对比', fontsize=14, pad=20)
        ax2.set_ylabel('覆盖率 (%)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, percentage in zip(bars2, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('mesh_renderer_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ 保存统计图表: mesh_renderer_statistics.png")
        
    except Exception as e:
        print(f"❌ 创建统计图表失败: {e}")

if __name__ == "__main__":
    print("🚀 开始mesh渲染器演示")
    demo_mesh_renderer()
    print("🎉 演示完成!")
