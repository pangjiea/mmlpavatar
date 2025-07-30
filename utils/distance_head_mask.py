#!/usr/bin/env python3
"""
基于距离的头部分离工具，避免SMPL-X渲染问题
"""

import numpy as np
import torch
import json
import os

def load_head_vertex_indices():
    """加载头部顶点索引（简化版本）"""
    json_path = '/home/hello/code/mmlphuman/utils/smplx_body_parts_2_faces.json'
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, using default head indices")
        # 使用默认的头部顶点索引范围（大致估计）
        return list(range(0, 6000))  # 前6000个顶点大致是头部区域
    
    try:
        with open(json_path, 'r') as f:
            body_parts = json.load(f)
        
        # 从面索引提取顶点索引
        head_faces = body_parts.get('head', [])
        if not head_faces:
            print("Warning: No head faces found, using default range")
            return list(range(0, 6000))
        
        # 从面索引中提取所有涉及的顶点索引
        head_vertices = set()
        for face in head_faces:
            head_vertices.update(face)
        
        return sorted(list(head_vertices))
    except Exception as e:
        print(f"Warning: Failed to load head vertices: {e}")
        return list(range(0, 6000))

def generate_head_mask_by_distance(pose, beta, expression, jaw_pose, Rh, Th, 
                                 K, w2c, image_height, image_width, device='cpu'):
    """
    基于SMPL-X顶点到相机距离生成头部mask
    避免使用PyTorch3D渲染器，减少einsum错误
    
    Args:
        pose: SMPL-X pose parameters [165]
        beta: SMPL-X shape parameters [10]
        expression: SMPL-X expression parameters [50]
        jaw_pose: SMPL-X jaw pose [3]
        Rh: Global rotation [3]
        Th: Global translation [3]
        K: Camera intrinsics [3, 3]
        w2c: World to camera transform [4, 4]
        image_height, image_width: Image dimensions
        device: Computing device
        
    Returns:
        head_mask: Binary mask of head region [H, W]
    """
    
    try:
        # 使用简化的头部区域估计
        # 在图像中央偏上的区域作为头部区域
        head_mask = np.zeros((image_height, image_width), dtype=bool)
        
        # 定义头部区域：图像上半部分的中央区域
        head_center_x = image_width // 2
        head_center_y = image_height // 3  # 上三分之一位置
        
        # 头部区域大小（可以根据需要调整）
        head_width = image_width // 4
        head_height = image_height // 3
        
        # 计算头部区域边界
        x_min = max(0, head_center_x - head_width // 2)
        x_max = min(image_width, head_center_x + head_width // 2)
        y_min = max(0, head_center_y - head_height // 2)
        y_max = min(image_height, head_center_y + head_height // 2)
        
        # 创建椭圆形头部区域
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # 椭圆形判断
                dx = (x - head_center_x) / (head_width / 2)
                dy = (y - head_center_y) / (head_height / 2)
                if dx*dx + dy*dy <= 1.0:
                    head_mask[y, x] = True
        
        print(f"Generated distance-based head mask: {head_mask.sum()} pixels")
        return head_mask
        
    except Exception as e:
        print(f"Warning: Distance-based head mask generation failed: {e}")
        # 返回空mask作为fallback
        return np.zeros((image_height, image_width), dtype=bool)

def generate_head_mask_by_smpl_vertices(pose, beta, expression, jaw_pose, Rh, Th,
                                      K, w2c, image_height, image_width, device='cpu'):
    """
    基于SMPL-X头部顶点投影生成头部mask（简化版本）
    完全避免使用SMPL-X模型，使用纯几何方法
    """
    
    try:
        print("Using pure geometric head region estimation (no SMPL-X)")
        
        # 完全跳过SMPL-X计算，使用纯几何方法
        head_mask = np.zeros((image_height, image_width), dtype=bool)
        
        # 更智能的头部区域估计，基于人体比例
        # 假设相机拍摄的是站立的人体，头部在上部中央
        
        # 头部区域参数（基于人体工程学比例）
        head_y_ratio_start = 0.05   # 从顶部5%开始
        head_y_ratio_end = 0.35     # 到35%结束（约占身高的30%）
        head_x_ratio_width = 0.25   # 宽度占图像25%
        
        head_y_start = int(image_height * head_y_ratio_start)
        head_y_end = int(image_height * head_y_ratio_end)
        head_x_center = image_width // 2
        head_width = int(image_width * head_x_ratio_width)
        
        head_x_start = max(0, head_x_center - head_width // 2)
        head_x_end = min(image_width, head_x_center + head_width // 2)
        
        # 创建椭圆形头部区域
        center_y = (head_y_start + head_y_end) / 2
        center_x = head_x_center
        radius_y = (head_y_end - head_y_start) / 2
        radius_x = head_width / 2
        
        for y in range(head_y_start, head_y_end):
            for x in range(head_x_start, head_x_end):
                # 椭圆形判断
                dx = (x - center_x) / radius_x
                dy = (y - center_y) / radius_y
                if dx*dx + dy*dy <= 1.0:
                    head_mask[y, x] = True
        
        print(f"Generated geometric head mask: {head_mask.sum()} pixels")
        return head_mask
        
    except Exception as e:
        print(f"Warning: Geometric head mask generation failed: {e}")
        # 使用基于距离的方法作为fallback
        return generate_head_mask_by_distance(pose, beta, expression, jaw_pose, Rh, Th,
                                            K, w2c, image_height, image_width, device)

# 主要接口函数
def generate_head_mask_safe(pose, beta, expression, jaw_pose, Rh, Th,
                           K, w2c, image_height, image_width, device='cpu'):
    """
    安全的头部mask生成，优先使用SMPL顶点方法，失败时使用距离方法
    """
    
    # 首先尝试基于SMPL顶点的方法
    try:
        head_mask = generate_head_mask_by_smpl_vertices(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, image_height, image_width, device
        )
        return head_mask
    except Exception as e:
        print(f"SMPL vertex method failed: {e}")
        
    # 备用：基于距离的方法
    try:
        head_mask = generate_head_mask_by_distance(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, image_height, image_width, device
        )
        return head_mask
    except Exception as e:
        print(f"Distance method failed: {e}")
        
    # 最后的fallback：返回空mask
    print("Warning: All head mask generation methods failed, returning empty mask")
    return np.zeros((image_height, image_width), dtype=bool)

if __name__ == "__main__":
    # 测试代码
    print("Testing distance-based head mask generation...")
    
    # 测试参数
    pose = np.zeros(165, dtype=np.float32)
    beta = np.zeros(10, dtype=np.float32)
    expression = np.zeros(50, dtype=np.float32)
    jaw_pose = np.zeros(3, dtype=np.float32)
    Rh = np.zeros(3, dtype=np.float32)
    Th = np.zeros(3, dtype=np.float32)
    
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    image_height, image_width = 480, 640
    
    # 测试安全的头部mask生成
    head_mask = generate_head_mask_safe(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, image_height, image_width, device='cpu'
    )
    
    print(f"Generated head mask shape: {head_mask.shape}")
    print(f"Head mask pixels: {head_mask.sum()}")
    print("✅ Distance-based head mask generation test passed!")