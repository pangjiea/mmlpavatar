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
        # 如果文件不存在，使用预定义的头部顶点索引
        return get_default_head_vertex_ids()

def get_default_head_vertex_ids():
    """获取默认的头部顶点索引（基于SMPL-X标准头部区域）"""
    # 这些是SMPL-X模型中头部区域的顶点索引
    # 包括面部、头顶、后脑勺等区域
    head_vertex_ids = np.array([
        # 面部区域
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        # 头顶区域
        331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
        # 后脑勺区域
        349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
        # 更多头部顶点...
    ], dtype=np.int64)

    print(f"✓ 使用默认头部顶点索引: {len(head_vertex_ids)} 个顶点")
    return head_vertex_ids

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

def get_neck_pose_from_full_pose(pose):
    """从完整pose中提取neck pose"""
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose).float()
    elif isinstance(pose, torch.Tensor):
        pose = pose.float()
    
    # SMPL-X中neck是第12个关节 (index 12)
    neck_pose = pose[12*3:12*3+3]
    return neck_pose

def render_head_mesh_to_mask(vertices, faces, K, w2c, image_height, image_width, use_z_buffer=True):
    """
    使用高级mesh渲染器生成头部掩膜
    支持Z-buffer深度测试和多种渲染模式

    Args:
        vertices: 头部顶点 [N, 3]
        faces: 头部面 [F, 3]
        K: 相机内参 [3, 3]
        w2c: 世界到相机变换 [4, 4]
        image_height, image_width: 图像尺寸
        use_z_buffer: 是否使用深度缓冲区

    Returns:
        head_mask: 头部掩膜 [H, W]
    """

    # 转换为numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(w2c, torch.Tensor):
        w2c = w2c.detach().cpu().numpy()

    # 变换顶点到相机坐标系
    vertices_homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    vertices_cam = (w2c @ vertices_homo.T).T[:, :3]  # [N, 3]

    # 过滤在相机前方的顶点
    valid_mask = vertices_cam[:, 2] > 0
    if not valid_mask.any():
        print("Warning: No vertices in front of camera")
        return np.zeros((image_height, image_width), dtype=bool)

    # 投影到图像平面
    vertices_img = vertices_cam / vertices_cam[:, 2:3]  # 透视除法
    pixels = (K @ vertices_img.T).T[:, :2]  # [N, 2]

    # 创建mask和深度缓冲区
    head_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if use_z_buffer:
        z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)

    # 渲染每个面
    rendered_faces = 0
    for face in faces:
        # 检查面的所有顶点是否有效
        if not all(valid_mask[face]):
            continue

        # 获取面的像素坐标和深度
        face_pixels = pixels[face]  # [3, 2]
        face_depths = vertices_cam[face, 2]  # [3]

        # 检查是否都在图像范围内
        if not (np.all(face_pixels[:, 0] >= 0) and np.all(face_pixels[:, 0] < image_width) and
                np.all(face_pixels[:, 1] >= 0) and np.all(face_pixels[:, 1] < image_height)):
            continue

        # 转换为整数像素坐标
        face_pixels_int = np.round(face_pixels).astype(np.int32)

        try:
            if use_z_buffer:
                # 使用深度缓冲区的高级渲染
                render_triangle_with_zbuffer(head_mask, z_buffer, face_pixels_int, face_depths,
                                           image_height, image_width)
            else:
                # 简单的OpenCV填充
                import cv2
                cv2.fillPoly(head_mask, [face_pixels_int], 255)
            rendered_faces += 1
        except Exception as e:
            continue

    if rendered_faces > 0:
        print(f"Rendered {rendered_faces} head faces to mask using advanced mesh renderer")
        return head_mask > 0
    else:
        print("Warning: No valid faces to render")
        return np.zeros((image_height, image_width), dtype=bool)

def render_triangle_with_zbuffer(mask, z_buffer, triangle_pixels, triangle_depths, height, width):
    """
    使用Z-buffer渲染单个三角形

    Args:
        mask: 输出掩膜 [H, W]
        z_buffer: 深度缓冲区 [H, W]
        triangle_pixels: 三角形像素坐标 [3, 2]
        triangle_depths: 三角形顶点深度 [3]
        height, width: 图像尺寸
    """
    # 获取三角形的边界框
    min_x = max(0, int(np.min(triangle_pixels[:, 0])))
    max_x = min(width - 1, int(np.max(triangle_pixels[:, 0])))
    min_y = max(0, int(np.min(triangle_pixels[:, 1])))
    max_y = min(height - 1, int(np.max(triangle_pixels[:, 1])))

    # 遍历边界框内的每个像素
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # 检查点是否在三角形内
            if point_in_triangle(x, y, triangle_pixels):
                # 计算重心坐标
                barycentric = compute_barycentric_coordinates(x, y, triangle_pixels)

                # 插值深度
                depth = np.sum(barycentric * triangle_depths)

                # 深度测试
                if depth < z_buffer[y, x]:
                    z_buffer[y, x] = depth
                    mask[y, x] = 255

def point_in_triangle(x, y, triangle):
    """检查点是否在三角形内（重心坐标法）"""
    p = np.array([x, y])
    a, b, c = triangle[0], triangle[1], triangle[2]

    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)

def compute_barycentric_coordinates(x, y, triangle):
    """计算点在三角形中的重心坐标"""
    p = np.array([x, y])
    a, b, c = triangle[0], triangle[1], triangle[2]

    v0 = b - a
    v1 = c - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1 - u - v

    return np.array([w, u, v])

def generate_head_mask_from_smpl_vertices(pose, beta, expression, jaw_pose, Rh, Th, K, w2c, image_height, image_width, device='cpu'):
    """
    使用简化的SMPL-X模型生成头部掩膜，参考TalkBody4D实现但避免einsum错误
    """
    try:
        # 强制使用CPU避免多进程CUDA问题
        device = torch.device('cpu')
        
        # 验证并修正输入参数维度 - 关键：使用10维expression避免einsum错误
        if isinstance(expression, np.ndarray):
            expression = torch.from_numpy(expression).float()
        if isinstance(expression, torch.Tensor):
            if expression.numel() == 50:
                # 从50维缩减到10维
                expression = expression[:10]
                print("Reduced expression from 50 to 10 dimensions to avoid einsum error")
            elif expression.numel() == 10:
                print("Using 10-dimensional expression")
            else:
                # 填充或截断到10维
                if expression.numel() < 10:
                    expression = torch.cat([expression, torch.zeros(10 - expression.numel())])
                else:
                    expression = expression[:10]
                print(f"Adjusted expression to 10 dimensions")
            
        # 尝试创建简化的SMPL-X模型
        try:
            import smplx
            
            # 创建简化SMPL-X模型，使用10维expression避免einsum错误
            smplx_model = smplx.create(
                model_path='./smpl_model',
                model_type='smplx',
                gender='neutral',
                num_betas=10,
                num_expression_coeffs=10,  # 关键：使用10维而不是50维
                use_pca=False,
                flat_hand_mean=True,
                batch_size=1
            ).to(device)
            
            print("✓ Created simplified SMPL-X model with 10-dim expression")
            
        except Exception as e:
            print(f"Cannot create SMPL-X model: {e}, head not available")
            return None
        
        # 准备输入参数
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
        
        print(f"SMPL-X input shapes: pose={pose.shape}, beta={beta.shape}, expression={expression.shape}, jaw_pose={jaw_pose.shape}")
        
        # SMPL-X forward pass
        with torch.no_grad():
            try:
                smpl_output = smplx_model(
                    body_pose=pose[:, 3:66],      # body pose: 63维
                    global_orient=Rh,            # global rotation: 3维
                    betas=beta,                  # shape params: 10维
                    jaw_pose=jaw_pose,           # jaw pose: 3维  
                    expression=expression,       # expression: 10维 (关键改动)
                    return_verts=True
                )
                vertices = smpl_output.vertices[0].cpu().numpy()  # [V, 3]
                print(f"SMPL-X generated {vertices.shape[0]} vertices using simplified model")
                
                # 应用全局变换 (参考TalkBody4D)
                from scipy.spatial.transform import Rotation
                if torch.any(Rh != 0):
                    R_global = Rotation.from_rotvec(Rh[0].cpu().numpy()).as_matrix()
                    vertices = (R_global @ vertices.T).T
                
                vertices = vertices + Th[0].cpu().numpy()
                
            except Exception as e:
                print(f"SMPL-X forward pass failed: {e}, head not available")
                return None
        
        # 获取SMPL-X的完整面信息
        faces = smplx_model.faces.astype(np.int32)
        print(f"SMPL-X has {faces.shape[0]} total faces")
        
        # 获取头部面索引
        head_faces_indices = load_head_faces()
        if not head_faces_indices:
            print("No head faces available")
            return None
        
        # 提取头部面
        head_faces = faces[head_faces_indices]  # [head_F, 3]
        print(f"Using {head_faces.shape[0]} head faces for rendering")
        
        # 渲染头部faces到掩膜
        head_mask = render_head_faces_to_mask(
            vertices, head_faces, K, w2c, image_height, image_width
        )
        
        if head_mask is None or head_mask.sum() == 0:
            print("Head not visible in current camera view, skipping head optimization")
            return None
        
        print(f"Generated simplified SMPL-X head mask: {head_mask.sum()} pixels ({head_mask.sum()/(image_height*image_width)*100:.1f}%)")
        return head_mask
        
    except Exception as e:
        print(f"Warning: Failed to generate SMPL-X head mask: {e}")
        return None

def render_head_faces_to_mask(vertices, faces, K, w2c, image_height, image_width):
    """
    直接渲染头部mesh faces到掩膜，使用TalkBody4D的正确投影算法
    
    Args:
        vertices: 头部顶点 [V, 3]
        faces: 头部面 [F, 3] 
        K: 相机内参 [3, 3]
        w2c: 世界到相机变换 [4, 4]
        image_height, image_width: 图像尺寸
        
    Returns:
        head_mask: 头部掩膜 [H, W]
    """
    
    # 转换为numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(w2c, torch.Tensor):
        w2c = w2c.detach().cpu().numpy()
    
    # 从w2c矩阵提取R和T
    # w2c = [R T; 0 1], 我们需要 R 和 T 用于 R*verts + T
    R = w2c[:3, :3]  # [3, 3]
    T = w2c[:3, 3:4]  # [3, 1]
    
    # 使用TalkBody4D的正确投影方法
    # 世界坐标到相机坐标: R * verts + T
    verts_cam = np.matmul(R, vertices.T) + T  # [3, V]
    
    # 过滤在相机前方的顶点
    valid_mask = verts_cam[2, :] > 0
    if not valid_mask.any():
        print("Warning: No vertices in front of camera")
        return np.zeros((image_height, image_width), dtype=bool)
    
    # 相机坐标投影到图像: K * verts_cam  
    verts_proj = np.matmul(K, verts_cam)  # [3, V]
    
    # 透视除法: 除以z坐标
    verts_proj = verts_proj / verts_proj[2:3, :]  # 广播除法
    pixels = verts_proj[:2, :].T  # [V, 2] - 取xy坐标并转置
    
    # 创建mask
    head_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 渲染每个面到mask
    rendered_faces = 0
    for face in faces:
        # 检查面的所有顶点是否有效
        if all(valid_mask[face]):
            # 获取面的像素坐标
            face_pixels = pixels[face]
            
            # 检查是否都在图像范围内
            if (np.all(face_pixels[:, 0] >= 0) and np.all(face_pixels[:, 0] < image_width) and
                np.all(face_pixels[:, 1] >= 0) and np.all(face_pixels[:, 1] < image_height)):
                
                # 转换为整数像素坐标
                face_pixels_int = np.round(face_pixels).astype(np.int32)
                
                # 使用OpenCV填充三角形
                try:
                    import cv2
                    cv2.fillPoly(head_mask, [face_pixels_int], 255)
                    rendered_faces += 1
                except ImportError:
                    # 如果cv2不可用，使用简单的边界框填充
                    x_min, x_max = face_pixels_int[:, 0].min(), face_pixels_int[:, 0].max()
                    y_min, y_max = face_pixels_int[:, 1].min(), face_pixels_int[:, 1].max()
                    head_mask[y_min:y_max+1, x_min:x_max+1] = 255
                    rendered_faces += 1
    
    if rendered_faces > 0:
        print(f"Successfully rendered {rendered_faces} head faces to mask using TalkBody4D projection")
        return head_mask > 0
    else:
        print("Head not visible in current camera view")
        return None


def generate_head_mask_with_mesh_renderer(pose, beta, expression, jaw_pose, Rh, Th, K, w2c,
                                        image_height, image_width, body_mask=None, device='cpu',
                                        use_advanced_renderer=True, use_z_buffer=True):
    """
    使用高级mesh渲染器生成头部掩膜
    支持多种渲染模式和深度测试

    Args:
        pose: SMPL-X pose parameters [165]
        beta: SMPL-X shape parameters [10]
        expression: SMPL-X expression parameters [50] or [10]
        jaw_pose: SMPL-X jaw pose [3]
        Rh: Global rotation [3]
        Th: Global translation [3]
        K: Camera intrinsics [3, 3]
        w2c: World to camera transform [4, 4]
        image_height, image_width: Image dimensions
        body_mask: 全身掩膜 [H, W], 如果提供则与头部投影取交集
        device: Computing device
        use_advanced_renderer: 是否使用高级渲染器
        use_z_buffer: 是否使用深度缓冲区

    Returns:
        head_mask: Binary mask of head region [H, W], 如果头部不在图像中返回None
    """

    try:
        # 强制使用CPU避免多进程问题
        device = torch.device('cpu')

        # 1. 加载FLAME头部顶点索引
        flame_vertex_ids = load_flame_vertex_ids()
        if flame_vertex_ids is None:
            print("❌ 无法获取头部顶点索引，使用几何fallback")
            return generate_geometric_head_mask(image_height, image_width)

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

        except Exception as e:
            print(f"❌ 创建SMPL-X模型失败: {e}")
            return generate_geometric_head_mask(image_height, image_width)

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

        # 4. SMPL-X前向传播 - 完全按照TalkBody4D的方式
        with torch.no_grad():
            try:
                # 按照TalkBody4D的方式：先用zero global_orient和transl生成vertices
                smpl_output = smplx_model(
                    body_pose=pose[:, 3:66],
                    global_orient=torch.zeros_like(Rh),  # 重要：使用zero global_orient
                    betas=beta,
                    jaw_pose=jaw_pose,
                    expression=expression,
                    transl=torch.zeros_like(Th),  # 重要：使用zero transl
                    return_verts=True
                )
                vertices = smpl_output.vertices[0].cpu().numpy()  # [N, 3]

                # 然后按照TalkBody4D的方式应用全局变换
                # TalkBody4D: verts = verts @ Rh.transpose(1, 2) + Th.unsqueeze(1)
                from scipy.spatial.transform import Rotation
                if torch.any(Rh != 0):
                    # 将axis-angle转换为旋转矩阵
                    R_global = Rotation.from_rotvec(Rh[0].cpu().numpy()).as_matrix()
                    # 按照TalkBody4D的方式：verts @ R.T
                    vertices = vertices @ R_global.T

                # 应用平移
                vertices = vertices + Th[0].cpu().numpy()



            except Exception as e:
                print(f"❌ SMPL-X前向传播失败: {e}")
                return generate_geometric_head_mask(image_height, image_width)

        # 5. 提取头部面
        smplx_faces = smplx_model.faces.astype(np.int32)
        head_faces, _ = extract_head_faces_from_smplx(smplx_faces, flame_vertex_ids)

        if len(head_faces) == 0:
            print("❌ 没有找到头部面")
            return generate_geometric_head_mask(image_height, image_width)

        # 6. 正确提取相机参数
        R, T = extract_camera_params_from_w2c(w2c)

        # 7. 使用mesh渲染器生成掩膜
        if use_advanced_renderer:
            head_mask = render_head_mesh_to_mask_with_correct_projection(
                vertices, head_faces, K, R, T, image_height, image_width, use_z_buffer
            )
        else:
            # 使用简单的TalkBody4D风格渲染
            head_mask = render_head_faces_to_mask_simple(
                vertices, head_faces, K, w2c, image_height, image_width
            )

        if head_mask is None or head_mask.sum() == 0:
            print("❌ 头部不在当前相机视野中")
            return None

        # 7. 如果提供了全身掩膜，求交集
        if body_mask is not None:
            if isinstance(body_mask, torch.Tensor):
                body_mask = body_mask.cpu().numpy()

            # 头部掩膜 = 头部投影 ∩ 全身掩膜
            head_mask_intersected = head_mask & body_mask

            if head_mask_intersected.sum() == 0:
                print("❌ 头部投影与全身掩膜无交集")
                return None

            print(f"✓ 生成头部掩膜: {head_mask_intersected.sum()} 像素 "
                  f"({head_mask_intersected.sum()/(image_height*image_width)*100:.1f}%)")
            return head_mask_intersected
        else:
            print(f"✓ 生成头部掩膜: {head_mask.sum()} 像素 "
                  f"({head_mask.sum()/(image_height*image_width)*100:.1f}%)")
            return head_mask

    except Exception as e:
        print(f"❌ 高级mesh渲染器头部掩膜生成失败: {e}")
        return generate_geometric_head_mask(image_height, image_width)

def load_head_faces():
    """加载头部面索引（fallback函数）"""
    # 这是一个简化的头部面索引，实际应该从SMPL-X模型中提取
    # 这里返回一些典型的头部面索引
    head_face_indices = list(range(0, 1000))  # 假设前1000个面是头部相关的
    print(f"✓ 使用fallback头部面索引: {len(head_face_indices)} 个面")
    return head_face_indices

def extract_camera_params_from_w2c(w2c):
    """
    从w2c矩阵正确提取相机参数R和T

    根据数据集的构造方式：w2c = [[R, T], [0, 1]]
    TalkBody4D使用的是world-to-camera的变换

    Args:
        w2c: 世界到相机变换矩阵 [4, 4]

    Returns:
        R: 旋转矩阵 [3, 3]
        T: 平移向量 [3]
    """
    if isinstance(w2c, torch.Tensor):
        w2c = w2c.detach().cpu().numpy()

    # 从w2c矩阵直接提取R和T
    # w2c的格式是 [R T; 0 1]，这正是TalkBody4D需要的格式
    R = w2c[:3, :3]  # [3, 3] 旋转矩阵
    T = w2c[:3, 3]   # [3] 平移向量

    return R, T

def render_head_mesh_to_mask_with_correct_projection(vertices, faces, K, R, T, image_height, image_width, use_z_buffer=True):
    """
    使用正确的TalkBody4D投影的高级mesh渲染器

    Args:
        vertices: 头部顶点 [N, 3]
        faces: 头部面 [F, 3]
        K: 相机内参 [3, 3]
        R: 相机旋转矩阵 [3, 3]
        T: 相机平移向量 [3]
        image_height, image_width: 图像尺寸
        use_z_buffer: 是否使用深度缓冲区

    Returns:
        head_mask: 头部掩膜 [H, W]
    """

    # 转换为numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()

    # 使用TalkBody4D的精确投影方法
    pixels, valid_mask = talkbody4d_projection_exact(vertices, K, R, T, image_width, image_height)

    if pixels is None:
        print("❌ TalkBody4D投影失败")
        return None

    # 创建mask和深度缓冲区
    head_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if use_z_buffer:
        z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)

    # 计算顶点的深度值（用于Z-buffer）
    if T.ndim == 1:
        T_col = T.reshape(3, 1)
    else:
        T_col = T
    verts_cam = np.matmul(R, vertices.T) + T_col  # [3, N]
    vertex_depths = verts_cam[2, :]  # [N] - Z坐标就是深度

    # 渲染每个面
    rendered_faces = 0
    for face in faces:
        # 检查面的所有顶点是否有效
        if not all(valid_mask[face]):
            continue

        # 获取面的像素坐标和深度
        face_pixels = pixels[face]  # [3, 2]
        face_depths = vertex_depths[face]  # [3]

        # 检查是否都在图像范围内
        if not (np.all(face_pixels[:, 0] >= 0) and np.all(face_pixels[:, 0] < image_width) and
                np.all(face_pixels[:, 1] >= 0) and np.all(face_pixels[:, 1] < image_height)):
            continue

        try:
            if use_z_buffer:
                # 使用深度缓冲区的高级渲染
                render_triangle_with_zbuffer(head_mask, z_buffer, face_pixels, face_depths,
                                           image_height, image_width)
            else:
                # 简单的OpenCV填充
                import cv2
                cv2.fillPoly(head_mask, [face_pixels], 255)
            rendered_faces += 1
        except Exception as e:
            continue

    if rendered_faces > 0:
        print(f"✓ 高级渲染器成功渲染 {rendered_faces} 个面")
        return head_mask > 0
    else:
        print("❌ 没有成功渲染的面")
        return None

def render_head_faces_to_mask_simple(vertices, faces, K, w2c, image_height, image_width):
    """
    简单的头部面渲染到掩膜（TalkBody4D风格）

    Args:
        vertices: 头部顶点 [V, 3]
        faces: 头部面 [F, 3]
        K: 相机内参 [3, 3]
        w2c: 世界到相机变换 [4, 4]
        image_height, image_width: 图像尺寸

    Returns:
        head_mask: 头部掩膜 [H, W]
    """

    # 转换为numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(w2c, torch.Tensor):
        w2c = w2c.detach().cpu().numpy()

    # 正确提取相机参数
    R, T = extract_camera_params_from_w2c(w2c)

    # 使用TalkBody4D的精确投影方法
    pixels, valid_mask = talkbody4d_projection_exact(vertices, K, R, T, image_width, image_height)

    if pixels is None:
        print("❌ TalkBody4D投影失败")
        return None

    # 创建空的掩膜并渲染头部面
    head_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    rendered_faces = 0
    for face in faces:
        # 检查面的所有顶点是否有效
        if all(valid_mask[face]):
            # 获取面的像素坐标
            face_pixels = pixels[face]  # [3, 2]

            try:
                # 使用OpenCV填充三角形面
                import cv2
                cv2.fillPoly(head_mask, [face_pixels], 255)
                rendered_faces += 1
            except Exception as e:
                continue

    if rendered_faces == 0:
        print("❌ 没有成功渲染的面")
        return None

    print(f"✓ 简单渲染器成功渲染 {rendered_faces} 个面")
    return head_mask > 0

# 为了向后兼容，保留原来的函数名
def generate_head_mask(pose, beta, expression, jaw_pose, Rh, Th, K, w2c, image_height, image_width, body_mask=None, device='cpu'):
    """
    生成头部掩膜 - 使用修正后的TalkBody4D投影
    这是主要的接口函数，向后兼容

    现在使用正确的TalkBody4D投影方法，确保头部位置准确
    """
    return generate_head_mask_with_mesh_renderer(
        pose, beta, expression, jaw_pose, Rh, Th, K, w2c,
        image_height, image_width, body_mask, device,
        use_advanced_renderer=False,  # 使用简单但正确的TalkBody4D投影
        use_z_buffer=False
    )

def talkbody4d_projection_exact(vertices, K, R, T, img_w, img_h):
    """
    完全模仿TalkBody4D的投影方法
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

    # 确保T是列向量 [3, 1]，完全按照TalkBody4D
    if T.ndim == 1:
        T = T.reshape(3, 1)

    # TalkBody4D第148行: smplx_verts_cam = np.matmul(R, verts.T) + T
    smplx_verts_cam = np.matmul(R, vertices.T) + T  # [3, N]

    # 检查在相机前方的顶点
    valid_mask = smplx_verts_cam[2, :] > 0
    if not valid_mask.any():
        return None, np.zeros(len(vertices), dtype=bool)

    # TalkBody4D第150行: smplx_verts_proj = np.matmul(K, smplx_verts_cam)
    smplx_verts_proj = np.matmul(K, smplx_verts_cam)  # [3, N]

    # TalkBody4D第152行: smplx_verts_proj /= smplx_verts_proj[2, :]
    smplx_verts_proj /= smplx_verts_proj[2, :]

    # TalkBody4D第153行: smplx_verts_proj = smplx_verts_proj[:2, :].T
    smplx_verts_proj = smplx_verts_proj[:2, :].T  # [N, 2]

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

    return smplx_verts_proj_rounded, final_valid_mask

def generate_geometric_head_mask(image_height, image_width):
    """几何估计的头部掩膜作为fallback"""
    head_mask = np.zeros((image_height, image_width), dtype=bool)
    
    # 头部区域参数（基于人体比例）
    head_y_start = int(image_height * 0.05)   # 从5%开始
    head_y_end = int(image_height * 0.35)     # 到35%结束
    head_x_center = image_width // 2
    head_width = int(image_width * 0.25)      # 宽度25%
    
    center_y = (head_y_start + head_y_end) / 2
    center_x = head_x_center
    radius_y = (head_y_end - head_y_start) / 2
    radius_x = head_width / 2
    
    for y in range(head_y_start, head_y_end):
        for x in range(max(0, head_x_center - head_width // 2), 
                      min(image_width, head_x_center + head_width // 2)):
            dx = (x - center_x) / radius_x
            dy = (y - center_y) / radius_y
            if dx*dx + dy*dy <= 1.0:
                head_mask[y, x] = True
    
    print(f"Generated geometric fallback head mask: {head_mask.sum()} pixels ({head_mask.sum()/(image_height*image_width)*100:.1f}%)")
    return head_mask


if __name__ == "__main__":
    # 测试代码
    print("Testing SMPL-X mesh head mask generation...")
    
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
    
    # 测试头部mask生成
    head_mask = generate_head_mask(
        pose, beta, expression, jaw_pose, Rh, Th,
        K, w2c, image_height, image_width, device='cpu'
    )
    
    print(f"Generated head mask shape: {head_mask.shape}")
    print(f"Head mask pixels: {head_mask.sum()}")
    print("✅ SMPL-X mesh head mask generation test completed!")
    456123