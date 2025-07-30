#!/usr/bin/env python3
"""
正确的头部掩膜生成工具
结合TalkBody4D的投影算法和GaussianSMPLXAvatars的头部提取方法
"""

import numpy as np
import torch
import json
import os
import cv2

def load_flame_vertex_ids():
    """从SMPL-X__FLAME_vertex_ids.npy文件加载FLAME头部顶点索引"""
    flame_ids_path = '/home/hello/code/mmlphuman/smpl_model/SMPL-X__FLAME_vertex_ids.npy'
    
    try:
        flame_vertex_ids = np.load(flame_ids_path)
        print(f"✓ 成功加载FLAME顶点索引: {len(flame_vertex_ids)} 个头部顶点")
        return flame_vertex_ids.astype(np.int64)
    except Exception as e:
        print(f"❌ 无法加载FLAME顶点索引: {e}")
        return None

def extract_head_faces_from_smplx(smplx_faces, flame_vertex_ids):
    """
    从SMPL-X面中提取头部面
    根据GaussianSMPLXAvatars的方法: 只有当面的所有顶点都属于头部时，才算头部面
    
    Args:
        smplx_faces: SMPL-X的所有面 [F, 3]
        flame_vertex_ids: FLAME头部顶点索引 [N]
        
    Returns:
        head_faces: 头部面 [head_F, 3]
        head_face_indices: 头部面在原faces中的索引
    """
    if isinstance(smplx_faces, torch.Tensor):
        smplx_faces = smplx_faces.cpu().numpy()
    
    # 创建头部顶点索引的set用于快速查找
    flame_vertex_set = set(flame_vertex_ids)
    
    # 找到所有顶点都在头部的面
    head_face_mask = np.all(np.isin(smplx_faces, flame_vertex_ids), axis=1)
    head_faces = smplx_faces[head_face_mask]
    head_face_indices = np.where(head_face_mask)[0]
    
    print(f"✓ 从 {len(smplx_faces)} 个总面中提取出 {len(head_faces)} 个头部面")
    return head_faces, head_face_indices

def project_vertices_talkbody4d_style(vertices, K, R, T):
    """
    使用TalkBody4D的正确投影方法
    
    Args:
        vertices: 3D顶点 [N, 3]
        K: 相机内参 [3, 3]  
        R: 相机旋转矩阵 [3, 3]
        T: 相机平移向量 [3, 1] 或 [3]
        
    Returns:
        pixels: 投影后的2D像素坐标 [N, 2]
        valid_mask: 有效顶点掩膜 [N]
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    
    # 确保T是列向量
    if T.ndim == 1:
        T = T.reshape(3, 1)
    
    # TalkBody4D的投影步骤:
    # 1. 世界坐标到相机坐标: R * verts.T + T
    verts_cam = np.matmul(R, vertices.T) + T  # [3, N]
    
    # 2. 过滤在相机前方的顶点 (z > 0)
    valid_mask = verts_cam[2, :] > 0
    
    if not valid_mask.any():
        print("⚠️ 没有顶点在相机前方")
        return np.zeros((len(vertices), 2)), valid_mask
    
    # 3. 相机坐标投影到图像: K * verts_cam
    verts_proj = np.matmul(K, verts_cam)  # [3, N]
    
    # 4. 透视除法: 除以z坐标  
    verts_proj = verts_proj / verts_proj[2:3, :]  # 广播除法 [3, N]
    
    # 5. 取xy坐标并转置为 [N, 2]
    pixels = verts_proj[:2, :].T  # [N, 2]
    
    return pixels, valid_mask

def render_head_faces_opencv(vertices, head_faces, K, R, T, image_height, image_width, body_mask=None):
    """
    使用OpenCV渲染头部面到掩膜
    结合TalkBody4D投影和GaussianSMPLXAvatars头部提取
    
    Args:
        vertices: SMPL-X顶点 [N, 3]
        head_faces: 头部面 [F, 3]
        K: 相机内参 [3, 3]
        R: 相机旋转矩阵 [3, 3] 
        T: 相机平移向量 [3, 1] 或 [3]
        image_height, image_width: 图像尺寸
        body_mask: 可选的全身掩膜，用于求交集
        
    Returns:
        head_mask: 头部掩膜 [H, W] 或 None
    """
    
    # 1. 使用TalkBody4D的投影方法
    pixels, valid_mask = project_vertices_talkbody4d_style(vertices, K, R, T)
    
    if not valid_mask.any():
        print("⚠️ 没有有效顶点可以投影")
        return None
    
    # 2. 创建空的掩膜
    head_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 3. 渲染每个头部面
    rendered_faces = 0
    for face in head_faces:
        # 检查面的所有顶点是否有效(在相机前方)
        if not all(valid_mask[face]):
            continue
            
        # 获取面的像素坐标
        face_pixels = pixels[face]  # [3, 2]
        
        # 检查是否都在图像范围内
        if (np.all(face_pixels[:, 0] >= 0) and np.all(face_pixels[:, 0] < image_width) and
            np.all(face_pixels[:, 1] >= 0) and np.all(face_pixels[:, 1] < image_height)):
            
            # 转换为整数像素坐标
            face_pixels_int = np.round(face_pixels).astype(np.int32)
            
            # 使用OpenCV填充三角形面
            cv2.fillPoly(head_mask, [face_pixels_int], 255)
            rendered_faces += 1
    
    if rendered_faces == 0:
        print("⚠️ 没有有效的头部面可以渲染")
        return None
    
    print(f"✓ 成功渲染 {rendered_faces} 个头部面")
    
    # 转换为bool掩膜
    head_mask_bool = head_mask > 0
    
    # 4. 如果提供了全身掩膜，求交集
    if body_mask is not None:
        if isinstance(body_mask, torch.Tensor):
            body_mask = body_mask.cpu().numpy()
        
        # 头部掩膜 = 头部投影 ∩ 全身掩膜
        head_mask_intersected = head_mask_bool & body_mask
        
        if head_mask_intersected.sum() == 0:
            print("⚠️ 头部投影与全身掩膜无交集")
            return None
        
        print(f"✓ 头部投影: {head_mask_bool.sum()} 像素")  
        print(f"✓ 与全身掩膜交集: {head_mask_intersected.sum()} 像素")
        return head_mask_intersected
    
    else:
        print(f"✓ 头部投影: {head_mask_bool.sum()} 像素")
        return head_mask_bool

def generate_correct_head_mask(pose, beta, expression, jaw_pose, Rh, Th, K, w2c, image_height, image_width, body_mask=None, device='cpu'):
    """
    正确的头部掩膜生成方法
    结合TalkBody4D投影算法和GaussianSMPLXAvatars头部提取
    
    Args:
        pose: SMPL-X pose parameters [165]
        beta: SMPL-X shape parameters [10]
        expression: SMPL-X expression parameters [50] 或 [10]
        jaw_pose: SMPL-X jaw pose [3]
        Rh: Global rotation [3]
        Th: Global translation [3] 
        K: Camera intrinsics [3, 3]
        w2c: World to camera transform [4, 4]
        image_height, image_width: Image dimensions
        body_mask: 可选的全身掩膜
        device: Computing device
        
    Returns:
        head_mask: Binary mask of head region [H, W] 或 None
    """
    
    try:
        # 强制使用CPU避免多进程问题
        device = torch.device('cpu')
        
        # 1. 加载FLAME头部顶点索引
        flame_vertex_ids = load_flame_vertex_ids()
        if flame_vertex_ids is None:
            print("❌ 无法获取头部顶点索引")
            return None
        
        # 2. 创建SMPL-X模型
        try:
            import smplx
            
            # 处理expression维度
            if isinstance(expression, np.ndarray):
                expression = torch.from_numpy(expression).float()
            if isinstance(expression, torch.Tensor):
                if expression.numel() == 50:
                    expression = expression[:10]
                    print("✓ 缩减expression从50维到10维")
                elif expression.numel() == 10:
                    print("✓ 使用10维expression")
                else:
                    if expression.numel() < 10:
                        expression = torch.cat([expression, torch.zeros(10 - expression.numel())])
                    else:
                        expression = expression[:10]
                    print("✓ 调整expression到10维")
            
            # 创建SMPL-X模型
            smplx_model = smplx.create(
                model_path='./smpl_model',
                model_type='smplx', 
                gender='neutral',
                num_betas=10,
                num_expression_coeffs=10,
                use_pca=False,
                flat_hand_mean=True,
                batch_size=1
            ).to(device)
            
            print("✓ 创建SMPL-X模型成功")
            
        except Exception as e:
            print(f"❌ 创建SMPL-X模型失败: {e}")
            return None
        
        # 3. 准备SMPL-X参数
        def prepare_param(param, target_device):
            if isinstance(param, np.ndarray):
                param = torch.from_numpy(param).float()
            param = param.to(target_device)
            return param.unsqueeze(0) if param.dim() == 1 else param
        
        pose = prepare_param(pose, device)
        beta = prepare_param(beta, device)
        expression = prepare_param(expression, device)
        jaw_pose = prepare_param(jaw_pose, device)
        Rh = prepare_param(Rh, device)
        Th = prepare_param(Th, device)
        
        # 4. SMPL-X前向传播
        with torch.no_grad():
            try:
                smpl_output = smplx_model(
                    body_pose=pose[:, 3:66],
                    global_orient=Rh,
                    betas=beta,
                    jaw_pose=jaw_pose,
                    expression=expression,
                    return_verts=True
                )
                vertices = smpl_output.vertices[0].cpu().numpy()  # [N, 3]
                print(f"✓ SMPL-X生成顶点: {vertices.shape}")
                
                # 应用全局变换
                from scipy.spatial.transform import Rotation
                if torch.any(Rh != 0):
                    R_global = Rotation.from_rotvec(Rh[0].cpu().numpy()).as_matrix()
                    vertices = (R_global @ vertices.T).T
                
                vertices = vertices + Th[0].cpu().numpy()
                
            except Exception as e:
                print(f"❌ SMPL-X前向传播失败: {e}")
                return None
        
        # 5. 提取头部面
        smplx_faces = smplx_model.faces.astype(np.int32)
        head_faces, head_face_indices = extract_head_faces_from_smplx(smplx_faces, flame_vertex_ids)
        
        if len(head_faces) == 0:
            print("❌ 没有找到头部面")
            return None
        
        # 6. 从w2c矩阵提取R和T
        if isinstance(w2c, torch.Tensor):
            w2c = w2c.detach().cpu().numpy()
        R = w2c[:3, :3]  # [3, 3]
        T = w2c[:3, 3]   # [3]
        
        # 7. 渲染头部掩膜
        head_mask = render_head_faces_opencv(
            vertices, head_faces, K, R, T, 
            image_height, image_width, body_mask
        )
        
        if head_mask is None:
            print("⚠️ 头部不在当前视角中，跳过头部优化")
            return None
        
        return head_mask
        
    except Exception as e:
        print(f"❌ 头部掩膜生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("测试正确的头部掩膜生成...")
    
    # 测试参数
    pose = np.zeros(165, dtype=np.float32)
    beta = np.zeros(10, dtype=np.float32)
    expression = np.zeros(50, dtype=np.float32)
    jaw_pose = np.zeros(3, dtype=np.float32)
    Rh = np.zeros(3, dtype=np.float32)
    Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    
    head_mask = generate_correct_head_mask(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, 480, 640, device='cpu'
    )
    
    if head_mask is not None:
        print(f"✅ 成功生成头部掩膜: {head_mask.sum()} 像素")
    else:
        print("❌ 头部掩膜生成失败")