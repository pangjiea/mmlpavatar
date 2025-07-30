#!/usr/bin/env python3
"""
测试新的正确头部掩膜生成，验证投影准确性
"""

import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('/home/hello/code/mmlphuman')

from utils.correct_head_mask_utils import generate_correct_head_mask

def create_test_visualization(head_mask, image_height, image_width, save_path):
    """创建测试可视化"""
    
    # 创建测试图像
    test_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 128
    
    # 在头部区域添加颜色
    if head_mask is not None:
        test_image[head_mask] = [255, 100, 100]  # 红色头部区域
    
    # 保存图像
    cv2.imwrite(save_path, test_image)
    print(f"✓ 可视化保存到: {save_path}")

def test_different_poses():
    """测试不同姿态下的头部投影"""
    
    print("🔍 测试不同姿态的头部投影准确性...")
    
    # 基础参数
    beta = np.zeros(10, dtype=np.float32)
    expression = np.zeros(50, dtype=np.float32)
    jaw_pose = np.zeros(3, dtype=np.float32)
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    image_height, image_width = 480, 640
    
    test_cases = [
        {
            'name': '正面_无旋转',
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.zeros(3, dtype=np.float32),
            'Th': np.array([0.0, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '头部左转', 
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.array([0.0, 0.3, 0.0], dtype=np.float32),  # Y轴旋转
            'Th': np.array([0.0, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '头部右转',
            'pose': np.zeros(165, dtype=np.float32), 
            'Rh': np.array([0.0, -0.3, 0.0], dtype=np.float32),  # Y轴旋转
            'Th': np.array([0.0, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '头部上仰',
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.array([0.3, 0.0, 0.0], dtype=np.float32),  # X轴旋转
            'Th': np.array([0.0, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '头部下俯',
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.array([-0.3, 0.0, 0.0], dtype=np.float32),  # X轴旋转
            'Th': np.array([0.0, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '整体向左移动',
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.zeros(3, dtype=np.float32),
            'Th': np.array([-0.5, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '整体向右移动',
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.zeros(3, dtype=np.float32), 
            'Th': np.array([0.5, 0.0, 2.0], dtype=np.float32),
        },
        {
            'name': '张嘴表情',
            'pose': np.zeros(165, dtype=np.float32),
            'Rh': np.zeros(3, dtype=np.float32),
            'Th': np.array([0.0, 0.0, 2.0], dtype=np.float32),
            'expression': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8] + [0.0] * 42, dtype=np.float32),  # 张嘴
        }
    ]
    
    os.makedirs("/home/hello/code/mmlphuman/debug_correct_head_masks", exist_ok=True)
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 测试案例 {i+1}: {test_case['name']}")
        
        # 获取参数
        pose = test_case['pose']
        Rh = test_case['Rh']
        Th = test_case['Th']
        expression = test_case.get('expression', expression)
        
        # 生成头部掩膜
        head_mask = generate_correct_head_mask(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, image_height, image_width, device='cpu'
        )
        
        # 记录结果
        if head_mask is not None:
            pixel_count = head_mask.sum()
            coverage = pixel_count / (image_height * image_width) * 100
            results.append({
                'name': test_case['name'],
                'pixels': pixel_count,
                'coverage': coverage,
                'success': True
            })
            print(f"✅ 成功: {pixel_count} 像素 ({coverage:.1f}% 覆盖率)")
        else:
            results.append({
                'name': test_case['name'],
                'pixels': 0,
                'coverage': 0.0,
                'success': False
            })
            print(f"❌ 失败: 头部不在视角中")
        
        # 创建可视化
        save_path = f"/home/hello/code/mmlphuman/debug_correct_head_masks/test_{i+1:02d}_{test_case['name'].replace(' ', '_')}.png"
        create_test_visualization(head_mask, image_height, image_width, save_path)
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 投影准确性测试总结")
    print("="*60)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ 成功案例: {len(successful_tests)}/{len(results)}")
    print(f"❌ 失败案例: {len(failed_tests)}/{len(results)}")
    
    if successful_tests:
        print(f"\n🎯 成功案例统计:")
        for result in successful_tests:
            print(f"  {result['name']}: {result['pixels']} 像素 ({result['coverage']:.1f}%)")
        
        avg_pixels = np.mean([r['pixels'] for r in successful_tests])
        avg_coverage = np.mean([r['coverage'] for r in successful_tests])
        print(f"\n📈 平均统计:")
        print(f"  平均像素数: {avg_pixels:.0f}")
        print(f"  平均覆盖率: {avg_coverage:.1f}%")
    
    if failed_tests:
        print(f"\n⚠️ 失败案例:")
        for result in failed_tests:
            print(f"  {result['name']}: 头部不在视角中")
    
    return results

def test_with_body_intersection():
    """测试与全身掩膜的交集"""
    
    print("\n🔍 测试头部掩膜与全身掩膜交集...")
    
    # 创建模拟的全身掩膜
    image_height, image_width = 480, 640
    body_mask = np.zeros((image_height, image_width), dtype=bool)
    
    # 创建一个人形的全身掩膜
    # 头部区域 (10%-30% height)
    head_y_start = int(image_height * 0.1)
    head_y_end = int(image_height * 0.3)
    head_x_center = image_width // 2
    head_width = int(image_width * 0.2)
    
    # 身体区域 (25%-80% height)
    body_y_start = int(image_height * 0.25)
    body_y_end = int(image_height * 0.8)
    body_width = int(image_width * 0.3)
    
    # 填充头部（椭圆）
    for y in range(head_y_start, head_y_end):
        for x in range(head_x_center - head_width//2, head_x_center + head_width//2):
            if 0 <= x < image_width:
                dx = (x - head_x_center) / (head_width/2)
                dy = (y - (head_y_start + head_y_end)/2) / ((head_y_end - head_y_start)/2)
                if dx*dx + dy*dy <= 1.0:
                    body_mask[y, x] = True
    
    # 填充身体（矩形）
    for y in range(body_y_start, body_y_end):
        for x in range(head_x_center - body_width//2, head_x_center + body_width//2):
            if 0 <= x < image_width:
                body_mask[y, x] = True
    
    print(f"✓ 创建模拟全身掩膜: {body_mask.sum()} 像素")
    
    # 测试参数
    pose = np.zeros(165, dtype=np.float32)
    beta = np.zeros(10, dtype=np.float32)
    expression = np.zeros(50, dtype=np.float32)
    jaw_pose = np.zeros(3, dtype=np.float32)
    Rh = np.zeros(3, dtype=np.float32)
    Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    
    # 不带body_mask的头部掩膜
    head_mask_only = generate_correct_head_mask(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, image_height, image_width, body_mask=None, device='cpu'
    )
    
    # 带body_mask的头部掩膜
    head_mask_intersected = generate_correct_head_mask(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, image_height, image_width, body_mask=body_mask, device='cpu'
    )
    
    print(f"\n📊 交集测试结果:")
    if head_mask_only is not None:
        print(f"  仅头部投影: {head_mask_only.sum()} 像素")
    else:
        print(f"  仅头部投影: None")
        
    if head_mask_intersected is not None:
        print(f"  与全身掩膜交集: {head_mask_intersected.sum()} 像素")
    else:
        print(f"  与全身掩膜交集: None")
    
    # 手动验证交集计算
    if head_mask_only is not None:
        manual_intersection = head_mask_only & body_mask
        print(f"  手动计算交集: {manual_intersection.sum()} 像素")
        
        if head_mask_intersected is not None:
            if head_mask_intersected.sum() == manual_intersection.sum():
                print("✅ 交集计算正确")
            else:
                print("❌ 交集计算有误")
    
    # 创建对比可视化
    comparison_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 50
    
    # 全身掩膜 - 蓝色
    comparison_image[body_mask] = [100, 100, 255]
    
    # 仅头部投影 - 绿色
    if head_mask_only is not None:
        comparison_image[head_mask_only] = [100, 255, 100]
    
    # 交集区域 - 红色（会覆盖之前的颜色）
    if head_mask_intersected is not None:
        comparison_image[head_mask_intersected] = [255, 100, 100]
    
    save_path = "/home/hello/code/mmlphuman/debug_correct_head_masks/body_intersection_test.png"
    cv2.imwrite(save_path, comparison_image)
    print(f"✓ 交集对比可视化保存到: {save_path}")

def main():
    print("🚀 开始正确头部掩膜投影准确性测试")
    print("="*60)
    
    # 测试1: 不同姿态下的投影
    pose_results = test_different_poses()
    
    # 测试2: 与全身掩膜的交集
    test_with_body_intersection()
    
    print("\n" + "="*60)
    print("🎉 所有测试完成!")
    print("="*60)
    
    # 验证关键指标
    successful_poses = len([r for r in pose_results if r['success']])
    total_poses = len(pose_results)
    
    print(f"\n📈 最终评估:")
    print(f"  姿态投影成功率: {successful_poses}/{total_poses} ({successful_poses/total_poses*100:.1f}%)")
    
    if successful_poses > 0:
        successful_results = [r for r in pose_results if r['success']]
        pixel_counts = [r['pixels'] for r in successful_results]
        
        print(f"  像素数范围: {min(pixel_counts)} - {max(pixel_counts)}")
        print(f"  像素数标准差: {np.std(pixel_counts):.0f}")
        
        # 判断是否合理
        if np.std(pixel_counts) < np.mean(pixel_counts) * 0.5:  # 标准差小于均值的50%
            print("✅ 不同姿态下的投影结果一致性良好")
        else:
            print("⚠️ 不同姿态下的投影结果变化较大")
    
    print(f"\n📁 所有可视化文件保存在: /home/hello/code/mmlphuman/debug_correct_head_masks/")

if __name__ == "__main__":
    main()