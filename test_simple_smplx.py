#!/usr/bin/env python3
"""
创建一个简化但有效的SMPL-X头部掩膜生成方法，参考TalkBody4D
"""

import numpy as np
import torch
import json
import os
import cv2

def create_simple_smplx_head_mask(pose, beta, expression, jaw_pose, Rh, Th, K, w2c, image_height, image_width):
    """
    使用简化的SMPL-X方法生成头部掩膜，避免einsum错误
    参考TalkBody4D的方法但简化参数
    """
    try:
        # 强制使用CPU避免多进程问题
        device = torch.device('cpu')
        
        # 创建简化的SMPL-X模型
        try:
            import smplx
            
            # 使用最简单的SMPL-X配置，避免复杂的expression处理
            smplx_model = smplx.create(
                model_path='./smpl_model',  # 假设smpl模型在这里
                model_type='smplx',
                gender='neutral',
                num_betas=10,
                num_expression_coeffs=10,  # 使用10维而不是50维
                use_pca=False,
                flat_hand_mean=True,
                batch_size=1
            ).to(device)
            
            print("✓ 创建简化SMPL-X模型成功")
            
        except Exception as e:
            print(f"无法创建SMPL-X模型: {e}")
            return None
        
        # 准备参数
        with torch.no_grad():
            # 转换参数为tensor
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float().to(device)
            if isinstance(beta, np.ndarray):
                beta = torch.from_numpy(beta).float().to(device)
            if isinstance(expression, np.ndarray):
                expression = torch.from_numpy(expression).float().to(device)
            if isinstance(jaw_pose, np.ndarray):
                jaw_pose = torch.from_numpy(jaw_pose).float().to(device)
            if isinstance(Rh, np.ndarray):
                Rh = torch.from_numpy(Rh).float().to(device)
            if isinstance(Th, np.ndarray):
                Th = torch.from_numpy(Th).float().to(device)
            
            # 确保维度正确
            if pose.dim() == 1:
                pose = pose.unsqueeze(0)
            if beta.dim() == 1:
                beta = beta.unsqueeze(0)
            if expression.dim() == 1:
                expression = expression.unsqueeze(0)
            if jaw_pose.dim() == 1:
                jaw_pose = jaw_pose.unsqueeze(0)
            if Rh.dim() == 1:
                Rh = Rh.unsqueeze(0)
            if Th.dim() == 1:
                Th = Th.unsqueeze(0)
            
            # 使用更简单的expression（只用前10维）
            if expression.shape[1] > 10:
                expression = expression[:, :10]
            elif expression.shape[1] < 10:
                # 填充到10维
                padding = torch.zeros(expression.shape[0], 10 - expression.shape[1]).to(device)
                expression = torch.cat([expression, padding], dim=1)
            
            print(f"参数形状: pose={pose.shape}, beta={beta.shape}, expr={expression.shape}, jaw={jaw_pose.shape}")
            
            # SMPL-X forward pass
            try:
                smpl_output = smplx_model(
                    body_pose=pose[:, 3:66],  # body pose
                    global_orient=Rh,        # global rotation
                    betas=beta,              # shape
                    jaw_pose=jaw_pose,       # jaw
                    expression=expression,    # expression (10维)
                    return_verts=True
                )
                
                vertices = smpl_output.vertices[0].cpu().numpy()  # [V, 3]
                print(f"✓ SMPL-X生成顶点: {vertices.shape}")
                
                # 应用全局变换 (参考TalkBody4D)
                from scipy.spatial.transform import Rotation
                if torch.any(Rh != 0):
                    R_global = Rotation.from_rotvec(Rh[0].cpu().numpy()).as_matrix()
                    vertices = (R_global @ vertices.T).T
                
                vertices = vertices + Th[0].cpu().numpy()
                
                # 获取头部面
                head_faces = load_head_faces_simple()
                if not head_faces:
                    print("无法加载头部面")
                    return None
                
                faces = smplx_model.faces.astype(np.int32)
                head_faces_selected = faces[head_faces]
                
                # 渲染头部
                head_mask = render_smplx_head_faces(
                    vertices, head_faces_selected, K, w2c, image_height, image_width
                )
                
                if head_mask is not None and head_mask.sum() > 0:
                    print(f"✓ 成功生成SMPL-X头部掩膜: {head_mask.sum()} 像素")
                    return head_mask
                else:
                    print("头部不在当前视角中")
                    return None
                    
            except Exception as e:
                print(f"SMPL-X forward pass失败: {e}")
                return None
                
    except Exception as e:
        print(f"创建SMPL-X头部掩膜失败: {e}")
        return None

def load_head_faces_simple():
    """简化的头部面加载"""
    try:
        json_path = '/home/hello/code/mmlphuman/utils/smplx_body_parts_2_faces.json'
        with open(json_path, 'r') as f:
            body_parts = json.load(f)
        return body_parts.get('head', [])
    except:
        return []

def render_smplx_head_faces(vertices, faces, K, w2c, image_height, image_width):
    """渲染SMPL-X头部面到掩膜"""
    try:
        # 转换为numpy
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
        if isinstance(w2c, torch.Tensor):
            w2c = w2c.detach().cpu().numpy()
        
        # 使用TalkBody4D的投影方法
        R = w2c[:3, :3]
        T = w2c[:3, 3:4]
        
        # 世界坐标到相机坐标: R * verts + T
        verts_cam = np.matmul(R, vertices.T) + T  # [3, V]
        
        # 相机坐标投影到图像: K * verts_cam
        verts_proj = np.matmul(K, verts_cam)  # [3, V]
        
        # 透视除法
        verts_proj = verts_proj / verts_proj[2:3, :]
        pixels = verts_proj[:2, :].T  # [V, 2]
        
        # 创建掩膜
        head_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # 渲染面
        rendered_faces = 0
        for face in faces:
            face_pixels = pixels[face]
            
            # 检查是否在图像范围内
            if (np.all(face_pixels[:, 0] >= 0) and np.all(face_pixels[:, 0] < image_width) and
                np.all(face_pixels[:, 1] >= 0) and np.all(face_pixels[:, 1] < image_height)):
                
                face_pixels_int = np.round(face_pixels).astype(np.int32)
                cv2.fillPoly(head_mask, [face_pixels_int], 255)
                rendered_faces += 1
        
        if rendered_faces > 0:
            print(f"渲染了 {rendered_faces} 个头部面")
            return head_mask > 0
        else:
            return None
            
    except Exception as e:
        print(f"渲染失败: {e}")
        return None

if __name__ == "__main__":
    print("测试简化SMPL-X头部掩膜生成...")
    
    # 测试参数
    pose = np.zeros(165, dtype=np.float32)
    beta = np.zeros(10, dtype=np.float32)
    expression = np.zeros(10, dtype=np.float32)  # 使用10维
    jaw_pose = np.zeros(3, dtype=np.float32)
    Rh = np.zeros(3, dtype=np.float32)
    Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    
    head_mask = create_simple_smplx_head_mask(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, 480, 640
    )
    
    if head_mask is not None:
        print(f"✓ 成功: {head_mask.sum()} 像素")
    else:
        print("头部不在视角中或生成失败")