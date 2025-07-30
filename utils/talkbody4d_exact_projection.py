#!/usr/bin/env python3
"""
完全模仿TalkBody4D的投影处理方法 - 最终正确版本
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
    """
    if isinstance(smplx_faces, torch.Tensor):
        smplx_faces = smplx_faces.cpu().numpy()
    
    # 找到所有顶点都在头部的面
    head_face_mask = np.all(np.isin(smplx_faces, flame_vertex_ids), axis=1)
    head_faces = smplx_faces[head_face_mask]
    head_face_indices = np.where(head_face_mask)[0]
    
    print(f"✓ 从 {len(smplx_faces)} 个总面中提取出 {len(head_faces)} 个头部面")
    return head_faces, head_face_indices

def talkbody4d_projection_exact(vertices, K, R, T, img_w, img_h):
    """
    完全模仿TalkBody4D的投影方法，包括所有细节
    
    对应TalkBody4D_utils.py的第148-157行
    """
    # 确保输入是numpy数组
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    
    print(f"TalkBody4D投影 - vertices shape: {vertices.shape}")
    print(f"TalkBody4D投影 - K shape: {K.shape}")
    print(f"TalkBody4D投影 - R shape: {R.shape}")
    print(f"TalkBody4D投影 - T shape: {T.shape}")
    
    # 确保T是列向量 [3, 1]，完全按照TalkBody4D
    if T.ndim == 1:
        T = T.reshape(3, 1)
    elif T.shape == (3, 3):  # 可能是w2c矩阵的T部分
        T = T.reshape(3, 1)
    
    # TalkBody4D第148行: smplx_verts_cam = np.matmul(R, verts.T) + T
    smplx_verts_cam = np.matmul(R, vertices.T) + T  # [3, N]
    
    print(f"相机坐标转换后: smplx_verts_cam shape = {smplx_verts_cam.shape}")
    print(f"前5个顶点的Z坐标: {smplx_verts_cam[2, :5]}")
    
    # 检查在相机前方的顶点
    valid_mask = smplx_verts_cam[2, :] > 0
    num_valid = valid_mask.sum()
    print(f"在相机前方的顶点数: {num_valid}/{len(vertices)}")
    
    if num_valid == 0:
        print("❌ 没有顶点在相机前方")
        return None, np.zeros(len(vertices), dtype=bool)
    
    # TalkBody4D第150行: smplx_verts_proj = np.matmul(K, smplx_verts_cam)
    smplx_verts_proj = np.matmul(K, smplx_verts_cam)  # [3, N]
    
    print(f"投影后: smplx_verts_proj shape = {smplx_verts_proj.shape}")
    
    # TalkBody4D第152行: smplx_verts_proj /= smplx_verts_proj[2, :]
    smplx_verts_proj /= smplx_verts_proj[2, :]
    
    # TalkBody4D第153行: smplx_verts_proj = smplx_verts_proj[:2, :].T
    smplx_verts_proj = smplx_verts_proj[:2, :].T  # [N, 2]
    
    print(f"透视除法后: smplx_verts_proj shape = {smplx_verts_proj.shape}")
    print(f"前5个顶点的像素坐标: {smplx_verts_proj[:5]}")
    
    # TalkBody4D第155行: smplx_verts_proj = np.round(smplx_verts_proj).astype(np.int32)
    smplx_verts_proj_rounded = np.round(smplx_verts_proj).astype(np.int32)
    
    # TalkBody4D第156-157行: clip到图像边界
    smplx_verts_proj_rounded[:, 0] = np.clip(smplx_verts_proj_rounded[:, 0], 0, img_w - 1)
    smplx_verts_proj_rounded[:, 1] = np.clip(smplx_verts_proj_rounded[:, 1], 0, img_h - 1)
    
    # 检查有效像素范围
    in_bounds_mask = (
        (smplx_verts_proj[:, 0] >= 0) & (smplx_verts_proj[:, 0] < img_w) &
        (smplx_verts_proj[:, 1] >= 0) & (smplx_verts_proj[:, 1] < img_h)
    )
    
    # 最终有效顶点：在相机前方且在图像范围内
    final_valid_mask = valid_mask & in_bounds_mask
    
    print(f"在图像范围内的顶点数: {final_valid_mask.sum()}/{len(vertices)}")
    
    return smplx_verts_proj_rounded, final_valid_mask

def render_head_faces_talkbody4d_style(vertices, head_faces, K, R, T, image_height, image_width, body_mask=None):
    """
    使用完全按照TalkBody4D的投影方法渲染头部面
    """
    
    # 使用TalkBody4D的精确投影方法
    pixels, valid_mask = talkbody4d_projection_exact(vertices, K, R, T, image_width, image_height)
    
    if pixels is None:
        print("❌ TalkBody4D投影失败")
        return None
    
    # 创建空的掩膜
    head_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 渲染每个头部面
    rendered_faces = 0
    valid_faces = 0
    
    for face in head_faces:
        # 检查面的所有顶点是否有效
        if all(valid_mask[face]):
            valid_faces += 1
            
            # 获取面的像素坐标
            face_pixels = pixels[face]  # [3, 2]
            
            try:
                # 使用OpenCV填充三角形面
                cv2.fillPoly(head_mask, [face_pixels], 255)
                rendered_faces += 1
            except Exception as e:
                print(f"渲染面时出错: {e}")
                continue
    
    print(f"✓ 有效面: {valid_faces}, 成功渲染面: {rendered_faces}")
    
    if rendered_faces == 0:
        print("❌ 没有成功渲染的面")
        return None
    
    # 转换为bool掩膜
    head_mask_bool = head_mask > 0
    
    # 如果提供了全身掩膜，求交集
    if body_mask is not None:
        if isinstance(body_mask, torch.Tensor):
            body_mask = body_mask.cpu().numpy()
        
        # 头部掩膜 = 头部投影 ∩ 全身掩膜
        head_mask_intersected = head_mask_bool & body_mask
        
        if head_mask_intersected.sum() == 0:
            print("❌ 头部投影与全身掩膜无交集")
            return None
        
        print(f"✓ 头部投影: {head_mask_bool.sum()} 像素")  
        print(f"✓ 与全身掩膜交集: {head_mask_intersected.sum()} 像素")
        return head_mask_intersected
    
    else:
        print(f"✓ 头部投影: {head_mask_bool.sum()} 像素")
        return head_mask_bool

def generate_head_mask_talkbody4d_exact(pose, beta, expression, jaw_pose, Rh, Th, K, w2c, image_height, image_width, body_mask=None, device='cpu'):
    """
    完全模仿TalkBody4D的头部掩膜生成方法
    """
    
    try:
        # 强制使用CPU避免多进程问题
        device = torch.device('cpu')
        
        print(f"🚀 开始TalkBody4D风格的头部掩膜生成")
        print(f"图像尺寸: {image_height} x {image_width}")
        
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
                
                # 应用全局变换 - 按照TalkBody4D的方式
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
        
        # 6. 从w2c矩阵提取R和T，完全按照TalkBody4D的格式
        if isinstance(w2c, torch.Tensor):
            w2c = w2c.detach().cpu().numpy()
        
        # TalkBody4D使用的是world-to-camera的R和T
        # w2c矩阵的格式是 [R T; 0 1]，其中R是旋转，T是平移
        R = w2c[:3, :3]  # [3, 3] 旋转矩阵
        T = w2c[:3, 3]   # [3] 平移向量
        
        print(f"相机参数 - R shape: {R.shape}, T shape: {T.shape}")
        print(f"T值: {T}")
        
        # 7. 使用完全模仿TalkBody4D的投影方法
        head_mask = render_head_faces_talkbody4d_style(
            vertices, head_faces, K, R, T, 
            image_height, image_width, body_mask
        )
        
        if head_mask is None:
            print("❌ TalkBody4D风格投影失败，头部不在当前视角中")
            return None
        
        print("✅ TalkBody4D风格头部掩膜生成成功")
        return head_mask
        
    except Exception as e:
        print(f"❌ TalkBody4D风格头部掩膜生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("测试TalkBody4D风格的头部掩膜生成...")
    
    # 测试参数
    pose = np.zeros(165, dtype=np.float32)
    beta = np.zeros(10, dtype=np.float32)
    expression = np.zeros(50, dtype=np.float32)
    jaw_pose = np.zeros(3, dtype=np.float32)
    Rh = np.zeros(3, dtype=np.float32)
    Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    
    head_mask = generate_head_mask_talkbody4d_exact(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, 480, 640, device='cpu'
    )
    
    if head_mask is not None:
        print(f"✅ TalkBody4D风格生成成功: {head_mask.sum()} 像素")
        
        # 保存可视化
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        test_image[head_mask] = [255, 100, 100]
        cv2.imwrite("/home/hello/code/mmlphuman/debug_talkbody4d_projection.png", test_image)
        print("✓ 可视化保存到: debug_talkbody4d_projection.png")
    else:
        print("❌ TalkBody4D风格生成失败")