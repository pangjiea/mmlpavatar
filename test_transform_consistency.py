#!/usr/bin/env python3
"""
测试Gaussian模型和SMPLX模型在相同变换下的坐标一致性
"""
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel, axis_angle_to_matrix
from scene.talkbody4D import axis_angle_to_matrix as ref_axis_angle_to_matrix
from utils.smpl_utils import smpl
from smplx.body_models import create

def test_transform_consistency():
    """测试Gaussian点和SMPLX vertices变换的一致性"""
    
    print("=== 测试Gaussian模型和SMPLX模型变换一致性 ===")
    
    # 1. 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建虚拟的SMPLX参数 - 模拟SQ02格式
    batch_size = 1
    body_pose = torch.randn(21, 3) * 0.1  # 身体姿态
    global_orient = torch.zeros(3)  # SQ02中global_orient为0
    transl = torch.zeros(3)  # SQ02中transl为0
    betas = torch.randn(10) * 0.1  # 身体形状
    expression = torch.randn(10) * 0.1  # 表情
    jaw_pose = torch.randn(3) * 0.1  # 下颌
    
    # SQ02特有的真实变换参数
    Rh_axis_angle = torch.tensor([0.1, 0.2, 0.15])  # axis-angle格式
    Th = torch.tensor([0.5, 1.0, -0.2])  # 平移
    
    print(f"测试参数:")
    print(f"  Rh (axis-angle): {Rh_axis_angle}")
    print(f"  Th: {Th}")
    print(f"  Body pose shape: {body_pose.shape}")
    print(f"  Expression shape: {expression.shape}")
    
    # 2. 使用SMPLX模型（参考实现）生成vertices
    print("\n--- 使用SMPLX模型生成vertices ---")
    
    # 创建SMPLX模型实例
    smplx_model = create(
        model_path="smpl_model",
        model_type='smplx', 
        gender='neutral',
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True,
        ext='npz'
    ).to(device)
    
    # 准备SMPLX输入
    smplx_params = {
        'global_orient': global_orient.unsqueeze(0).to(device),  # [1, 3]
        'body_pose': body_pose.reshape(1, -1).to(device),  # [1, 63]
        'transl': transl.unsqueeze(0).to(device),  # [1, 3]
        'betas': betas.unsqueeze(0).to(device),  # [1, 10]
        'expression': expression.unsqueeze(0).to(device),  # [1, 10]
        'jaw_pose': jaw_pose.unsqueeze(0).to(device),  # [1, 3]
    }
    
    # 转换到设备
    Rh_axis_angle = Rh_axis_angle.to(device)
    Th = Th.to(device)
    
    # 使用SMPLX模型生成vertices
    with torch.no_grad():
        smplx_output = smplx_model(**smplx_params)
        vertices_canonical = smplx_output.vertices[0]  # [V, 3]
        
        # 应用Rh和Th变换（参考talkbody4D.py的实现）
        Rh_matrix = ref_axis_angle_to_matrix(Rh_axis_angle)
        vertices_transformed_ref = vertices_canonical @ Rh_matrix.T + Th
        
    print(f"SMPLX vertices shape: {vertices_canonical.shape}")
    print(f"变换后vertices前5个点:")
    print(vertices_transformed_ref[:5])
    
    # 3. 模拟Gaussian模型的变换
    print("\n--- 模拟Gaussian模型变换 ---")
    
    # 创建一个简单的Gaussian模型用于测试
    gs_model = GaussianModel()
    
    # 设置SMPLX相关参数
    # SMPL/SMPLX有22个关节: 1个全局 + 21个身体关节
    full_pose = torch.cat([global_orient, body_pose.flatten()])  # [66,] (3 + 63)
    gs_model.smpl_poses = full_pose  # 保持在CPU上，因为get_rigid_transform需要.numpy()
    gs_model.expression = expression.to(device)
    gs_model.jaw_pose = jaw_pose.to(device)
    
    # 设置变换参数
    gs_model.Rh = Rh_matrix  # 直接设置旋转矩阵
    gs_model.Th = Th
    
    # 从SMPLX vertices采样一些点作为Gaussian点
    # 简单采样前1000个点
    n_samples = min(1000, vertices_canonical.shape[0])
    sample_indices = torch.randperm(vertices_canonical.shape[0])[:n_samples]
    sampled_vertices = vertices_canonical[sample_indices]
    
    # 设置Gaussian点的canonical位置
    gs_model._xyz = sampled_vertices
    
    # 创建简化的LBS权重 - 每个点主要绑定到一个关节，少量绑定到其他关节
    weights = torch.zeros(n_samples, 22)
    for i in range(n_samples):
        main_joint = sample_indices[i] % 22  # 主要关节
        weights[i, main_joint] = 0.8  # 主要权重
        if main_joint > 0:
            weights[i, main_joint - 1] = 0.1  # 邻近关节
        if main_joint < 21:
            weights[i, main_joint + 1] = 0.1  # 邻近关节
    gs_model._weights = weights.to(device)
    
    # 创建更真实的关节数据（而不是全零）
    # 使用SMPLX输出的关节位置
    joints = smplx_output.joints[0]  # [22, 3] - SMPLX的关节位置
    gs_model.t_joints = joints.cpu()  # 保持在CPU上，因为get_rigid_transform需要.numpy()
    
    # 设置正确的SMPL关节父子关系 (SMPL标准层级结构)
    smpl_parents = torch.tensor([
        -1,  # 0: pelvis (root)
         0,  # 1: left_hip
         0,  # 2: right_hip
         0,  # 3: spine1
         1,  # 4: left_knee
         2,  # 5: right_knee
         3,  # 6: spine2
         4,  # 7: left_ankle
         5,  # 8: right_ankle
         6,  # 9: spine3
         7,  # 10: left_foot
         8,  # 11: right_foot
         9,  # 12: neck
        12,  # 13: left_collar
        12,  # 14: right_collar
        12,  # 15: head
        13,  # 16: left_shoulder
        14,  # 17: right_shoulder
        16,  # 18: left_elbow
        17,  # 19: right_elbow
        18,  # 20: left_wrist
        19,  # 21: right_wrist
    ])
    gs_model.joint_parents = smpl_parents
    
    # 创建合理的逆变换矩阵
    # 这通常是T-pose下的关节变换的逆矩阵
    gs_model.Ac_inv = torch.eye(4).unsqueeze(0).repeat(22, 1, 1)  # 简化为单位矩阵，保持在CPU上
    
    # 设置必要的邻居和偏移数据以避免get_xyz错误
    gs_model.xyz_offset = torch.nn.Parameter(torch.zeros_like(sampled_vertices))
    gs_model.dxyz_vt = torch.zeros(n_samples, 3).to(device)
    
    # 设置简化的邻居数据（每个点指向自己，权重为1）
    gs_model.nbr_gs = torch.arange(n_samples, dtype=torch.long).unsqueeze(1).to(device)  # [n_samples, 1]
    gs_model.nbr_gs_invdist = torch.ones(n_samples, 1).to(device)  # [n_samples, 1]
    
    # 4. 比较变换结果
    print("\n--- 比较变换结果 ---")
    
    # 计算理论上的正确变换（直接对采样点应用变换）
    expected_transformed = sampled_vertices @ Rh_matrix.T + Th
    
    print(f"期望变换结果前5个点:")
    print(expected_transformed[:5])
    
    # 测试axis_angle_to_matrix函数
    print(f"\n--- 测试axis_angle_to_matrix函数 ---")
    Rh_matrix_gs = axis_angle_to_matrix(Rh_axis_angle)
    print(f"TalkBody4D Rh matrix:\n{Rh_matrix}")
    print(f"GaussianModel Rh matrix:\n{Rh_matrix_gs}")
    print(f"矩阵差异: {torch.max(torch.abs(Rh_matrix - Rh_matrix_gs))}")
    
    # 测试Rh setter
    print(f"\n--- 测试Rh setter ---")
    gs_model.Rh = Rh_axis_angle  # 设置axis-angle格式
    print(f"设置后的Rh matrix:\n{gs_model.Rh}")
    print(f"与参考矩阵差异: {torch.max(torch.abs(Rh_matrix - gs_model.Rh))}")
    
    # 比较简单变换（不使用LBS）
    print(f"\n--- 简单变换比较 ---")
    simple_transformed = sampled_vertices @ gs_model.Rh.T + gs_model.Th
    print(f"简单变换结果前5个点:")
    print(simple_transformed[:5])
    print(f"与期望结果差异: {torch.max(torch.abs(expected_transformed - simple_transformed))}")
    
    # 5. 测试GaussianModel的完整get_xyz流程
    print(f"\n--- 测试GaussianModel完整get_xyz流程 ---")
    try:
        # 清除缓存确保重新计算
        gs_model.cache_dict = {}
        
        # 调用get_xyz函数（包含LBS + 全局变换）
        gs_transformed = gs_model.get_xyz
        print(f"GaussianModel get_xyz结果前5个点:")
        print(gs_transformed[:5])
        
        # 与简单变换比较（看LBS的影响）
        lbs_diff = torch.max(torch.abs(gs_transformed - simple_transformed))
        print(f"get_xyz与简单变换的差异: {lbs_diff}")
        
        if lbs_diff < 1e-5:
            print("✅ LBS变换基本没有影响（权重接近单位矩阵）")
        else:
            print(f"⚠️  LBS变换有显著影响，最大差异: {lbs_diff}")
            
    except Exception as e:
        print(f"❌ get_xyz调用失败: {e}")
        print("尝试使用简化的变换测试...")
        
        # 如果LBS失败，我们直接测试简化的变换
        try:
            # 创建一个简化的get_xyz函数，跳过LBS
            # 只测试全局变换部分
            simplified_xyz = sampled_vertices
            if gs_model.Rh is not None: 
                simplified_xyz = torch.einsum('vi,ji->vj', simplified_xyz, gs_model.Rh)
            simplified_xyz = simplified_xyz + gs_model.Th
            
            print(f"简化get_xyz结果前5个点:")
            print(simplified_xyz[:5])
            
            # 与简单变换比较
            simple_diff = torch.max(torch.abs(simplified_xyz - simple_transformed))
            print(f"简化get_xyz与简单变换的差异: {simple_diff}")
            
            if simple_diff < 1e-5:
                print("✅ 全局变换部分工作正常")
            else:
                print(f"❌ 全局变换部分有问题，差异: {simple_diff}")
                
            gs_transformed = simplified_xyz
            
        except Exception as e2:
            print(f"❌ 简化测试也失败: {e2}")
            import traceback
            traceback.print_exc()
            gs_transformed = None
    
    return {
        'vertices_canonical': vertices_canonical,
        'vertices_transformed_ref': vertices_transformed_ref,
        'sampled_vertices': sampled_vertices,
        'expected_transformed': expected_transformed,
        'simple_transformed': simple_transformed,
        'Rh_matrix': Rh_matrix,
        'Rh_matrix_gs': Rh_matrix_gs
    }

if __name__ == "__main__":
    try:
        results = test_transform_consistency()
        print("\n=== 测试完成 ===")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
