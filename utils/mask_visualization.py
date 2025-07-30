#!/usr/bin/env python3
"""
头部mask可视化和保存工具
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

def save_head_mask_visualization(image, head_mask, save_path, title="Head Mask Visualization"):
    """
    保存头部mask和原图的对比可视化
    
    Args:
        image: 原图 [H, W, 3] 范围0-1 或 0-255
        head_mask: 头部mask [H, W] bool类型
        save_path: 保存路径
        title: 图片标题
    """
    
    # 确保图像是正确的格式
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 确保head_mask是bool类型
    if head_mask.dtype != bool:
        head_mask = head_mask.astype(bool)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # 1. 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. 头部mask
    axes[0, 1].imshow(head_mask, cmap='gray')
    axes[0, 1].set_title(f'Head Mask ({head_mask.sum()} pixels, {head_mask.sum()/(head_mask.shape[0]*head_mask.shape[1])*100:.1f}%)')
    axes[0, 1].axis('off')
    
    # 3. 头部区域（mask叠加在原图上）
    head_overlay = image.copy()
    head_overlay[head_mask] = head_overlay[head_mask] * 0.7 + np.array([255, 0, 0]) * 0.3  # 红色叠加
    axes[1, 0].imshow(head_overlay.astype(np.uint8))
    axes[1, 0].set_title('Head Region Overlay (Red)')
    axes[1, 0].axis('off')
    
    # 4. 头部分离效果
    head_only = np.zeros_like(image)
    head_only[head_mask] = image[head_mask]
    axes[1, 1].imshow(head_only)
    axes[1, 1].set_title('Head Region Only')
    axes[1, 1].axis('off')
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Head mask visualization saved to: {save_path}")

def save_head_mask_comparison(image, head_mask, body_mask=None, save_path=None, frame_id=0, cam_id=0):
    """
    保存头部和身体mask的对比
    
    Args:
        image: 原图 [H, W, 3]
        head_mask: 头部mask [H, W]
        body_mask: 身体mask [H, W] (可选)
        save_path: 保存路径
        frame_id: 帧ID
        cam_id: 相机ID
    """
    
    if save_path is None:
        save_path = f"debug_masks/frame_{frame_id:06d}_cam_{cam_id}.png"
    
    # 创建身体mask（如果没有提供）
    if body_mask is None:
        # 假设整个人体区域就是非头部区域（简化）
        body_mask = np.ones_like(head_mask, dtype=bool)
        body_mask[head_mask] = False
    
    # 确保图像格式正确
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    # 创建分离效果图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Head-Body Separation - Frame {frame_id}, Camera {cam_id}', fontsize=16)
    
    # 第一行：原图和mask
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(head_mask, cmap='Reds', alpha=0.8)
    axes[0, 1].set_title(f'Head Mask ({head_mask.sum()} pixels)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(body_mask, cmap='Blues', alpha=0.8)  
    axes[0, 2].set_title(f'Body Mask ({body_mask.sum()} pixels)')
    axes[0, 2].axis('off')
    
    # 第二行：分离效果
    # 头部区域
    head_only = np.zeros_like(image)
    head_only[head_mask] = image[head_mask]
    axes[1, 0].imshow(head_only)
    axes[1, 0].set_title('Head Region')
    axes[1, 0].axis('off')
    
    # 身体区域  
    body_only = np.zeros_like(image)
    body_only[body_mask] = image[body_mask]
    axes[1, 1].imshow(body_only)
    axes[1, 1].set_title('Body Region')
    axes[1, 1].axis('off')
    
    # 合并显示
    combined = image.copy()
    combined[head_mask] = combined[head_mask] * 0.7 + np.array([255, 0, 0]) * 0.3  # 头部红色
    combined[body_mask] = combined[body_mask] * 0.7 + np.array([0, 0, 255]) * 0.3  # 身体蓝色
    axes[1, 2].imshow(combined.astype(np.uint8))
    axes[1, 2].set_title('Combined (Head=Red, Body=Blue)')
    axes[1, 2].axis('off')
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Head-body comparison saved to: {save_path}")

def create_mask_overlay_video(image_dir, mask_dir, output_path, fps=10):
    """
    创建mask叠加的视频
    
    Args:
        image_dir: 原图目录
        mask_dir: mask目录  
        output_path: 输出视频路径
        fps: 帧率
    """
    
    import glob
    
    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                        glob.glob(os.path.join(image_dir, "*.png")))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # 读取第一张图像获取尺寸
    first_img = cv2.imread(image_files[0])
    height, width = first_img.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img_path in image_files:
        # 读取图像
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 构造对应的mask路径
        basename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, basename.replace('.jpg', '_mask.png').replace('.png', '_mask.png'))
        
        if os.path.exists(mask_path):
            # 读取mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask > 128  # 二值化
            
            # 创建叠加效果
            overlay = img_rgb.copy()
            overlay[mask] = overlay[mask] * 0.7 + np.array([255, 0, 0]) * 0.3
            
            # 转换回BGR并写入视频
            overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(overlay_bgr)
        else:
            # 如果没有mask，直接使用原图
            out.write(img)
    
    out.release()
    print(f"Mask overlay video saved to: {output_path}")

# 集成到数据集中的保存函数
def save_dataset_head_mask_sample(data_dict, save_dir, sample_id, max_samples=10):
    """
    从数据集中保存头部mask样本
    
    Args:
        data_dict: 数据集返回的字典，包含 'image', 'head_mask' 等
        save_dir: 保存目录
        sample_id: 样本ID
        max_samples: 最大保存样本数
    """
    
    # 限制保存的样本数量，避免磁盘空间问题
    if sample_id >= max_samples:
        return
    
    try:
        # 提取数据
        image = data_dict['image'].numpy()  # [3, H, W] 或 [H, W, 3]
        head_mask = data_dict['head_mask'].numpy()  # [H, W]
        frame_id = data_dict.get('frame_id', sample_id)
        cam_id = data_dict.get('cam_id', 0)
        
        # 确保图像格式正确
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)  # [3,H,W] -> [H,W,3]
        
        # 保存路径
        save_path = os.path.join(save_dir, f"sample_{sample_id:03d}_frame_{frame_id:06d}_cam_{cam_id}.png")
        
        # 保存可视化
        save_head_mask_visualization(image, head_mask, save_path, 
                                   title=f'Sample {sample_id} - Frame {frame_id}, Camera {cam_id}')
        
    except Exception as e:
        print(f"Failed to save head mask sample {sample_id}: {e}")

if __name__ == "__main__":
    # 测试代码
    print("Testing head mask visualization...")
    
    # 创建测试数据
    height, width = 480, 640
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 创建一个椭圆形的头部mask
    head_mask = np.zeros((height, width), dtype=bool)
    center_y, center_x = height // 4, width // 2
    radius_y, radius_x = height // 6, width // 8
    
    for y in range(height):
        for x in range(width):
            if ((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2 <= 1:
                head_mask[y, x] = True
    
    # 保存测试可视化
    save_head_mask_visualization(image, head_mask, "test_head_mask.png", "Test Head Mask")
    save_head_mask_comparison(image, head_mask, save_path="test_head_body_comparison.png")
    
    print("✅ Head mask visualization test completed!")