#!/usr/bin/env python3
"""
正确的Gaussian Splatting PLY导出脚本
基于标准3DGS格式，确保兼容性
"""

import os
import sys
import numpy as np
from argparse import ArgumentParser

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile package is required. Install with: pip install plyfile")
    sys.exit(1)


def save_correct_gaussian_ply(xyz, colors, opacities, scales, rotations, output_path):
    """
    保存正确格式的Gaussian Splatting PLY文件
    
    Args:
        xyz: [N, 3] 位置
        colors: [N, 3] RGB颜色 (0-1范围)
        opacities: [N] 透明度 (0-1范围)
        scales: [N, 3] 缩放
        rotations: [N, 4] 四元数 (w, x, y, z)
        output_path: 输出路径
    """
    
    # 确保数据类型和范围正确
    xyz = xyz.astype(np.float32)
    colors = np.clip(colors, 0, 1).astype(np.float32)
    opacities = np.clip(opacities, 0, 1).astype(np.float32)
    scales = scales.astype(np.float32)
    rotations = rotations.astype(np.float32)
    
    # 归一化四元数
    rotation_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotation_norms = np.maximum(rotation_norms, 1e-8)
    rotations = rotations / rotation_norms
    
    # 创建球谐系数 (DC分量)
    # 将RGB转换为球谐DC分量
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0
    
    # 创建零法向量
    normals = np.zeros_like(xyz)
    
    # 构建PLY属性
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    # 组装数据
    vertex_data = np.empty(len(xyz), dtype=dtype_list)
    
    # 位置和法向量
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]
    
    # 球谐DC分量
    vertex_data['f_dc_0'] = f_dc[:, 0]
    vertex_data['f_dc_1'] = f_dc[:, 1]
    vertex_data['f_dc_2'] = f_dc[:, 2]
    
    # 透明度
    vertex_data['opacity'] = opacities
    
    # 缩放
    vertex_data['scale_0'] = scales[:, 0]
    vertex_data['scale_1'] = scales[:, 1]
    vertex_data['scale_2'] = scales[:, 2]
    
    # 旋转四元数
    vertex_data['rot_0'] = rotations[:, 0]
    vertex_data['rot_1'] = rotations[:, 1]
    vertex_data['rot_2'] = rotations[:, 2]
    vertex_data['rot_3'] = rotations[:, 3]
    
    # 创建PLY元素
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # 保存文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([vertex_element]).write(output_path)
    
    print(f"保存PLY文件: {output_path}")
    print(f"  点数: {len(xyz)}")
    print(f"  位置范围: X[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
    print(f"  缩放范围: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  透明度范围: [{opacities.min():.3f}, {opacities.max():.3f}]")


def create_test_gaussian_data(n_points=1000):
    """创建测试用的Gaussian数据"""
    
    # 创建球形分布的点
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.5, 1.0, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    xyz = np.column_stack([x, y, z])
    
    # 基于位置的颜色
    colors = (xyz + 1) / 2  # 归一化到[0,1]
    
    # 合理的透明度
    opacities = np.random.uniform(0.5, 1.0, n_points)
    
    # 小的缩放值
    scales = np.random.uniform(0.001, 0.01, (n_points, 3))
    
    # 随机但归一化的四元数
    rotations = np.random.randn(n_points, 4)
    rotation_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / rotation_norms
    
    return xyz, colors, opacities, scales, rotations


def export_from_gaussian_model_corrected(model_dir, smpl_params_path, output_dir, frame_indices=[0]):
    """从Gaussian模型导出，使用正确的格式"""
    
    try:
        import torch
        from scene.gaussian_model import GaussianModel
        from utils.smpl_utils import init_smpl
        from omegaconf import OmegaConf
        
        print(f"加载Gaussian模型: {model_dir}")
        
        # 加载配置和模型
        config_path = os.path.join(model_dir, 'config.yaml')
        config = OmegaConf.load(config_path)
        
        smpl_pkl_path = config.smpl_pkl_path
        if not os.path.isabs(smpl_pkl_path):
            smpl_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), smpl_pkl_path)
        
        init_smpl(smpl_pkl_path)
        
        gaussians = GaussianModel()
        checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('chkpnt') and f.endswith('.pth')]
        checkpoint_files.sort(key=lambda x: int(x.replace('chkpnt', '').replace('.pth', '')))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        gaussians.restore(checkpoint)
        gaussians.is_test = True
        gaussians.prepare_test()
        
        # 加载SMPL参数
        smpl_data = np.load(smpl_params_path, allow_pickle=True)
        poses, transl, expressions, jaw_poses = gaussians._parse_smpl_params(smpl_data)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in frame_indices:
            print(f"\n处理帧 {frame_idx}...")
            
            # 设置帧参数
            gaussians._set_frame_params(
                poses[frame_idx], transl[frame_idx],
                expressions[frame_idx] if expressions is not None else None,
                jaw_poses[frame_idx] if jaw_poses is not None else None
            )
            
            with torch.no_grad():
                # 获取位置
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                
                # 获取颜色 (使用球谐函数)
                sh = gaussians.get_sh.detach().cpu().numpy()
                if sh.shape[1] > 0:
                    # 使用DC分量作为颜色
                    colors = sh[:, 0, :] + 0.5  # 球谐DC分量转RGB
                    colors = np.clip(colors, 0, 1)
                else:
                    colors = np.random.rand(len(xyz), 3)
                
                # 获取透明度并修正
                opacities = gaussians.get_opacity.detach().cpu().numpy()
                opacities = 1.0 / (1.0 + np.exp(-opacities))  # sigmoid激活
                
                # 获取缩放并修正
                scales = gaussians.get_cano_scaling.detach().cpu().numpy()
                scales = np.exp(scales)  # 从log空间转换
                scales = np.clip(scales, 1e-6, 0.005)  # 严格限制缩放
                
                # 获取旋转并修正
                rotations = gaussians.get_cano_rotation.detach().cpu().numpy()
                
                # 保存PLY
                output_path = os.path.join(output_dir, f"corrected_frame_{frame_idx:06d}.ply")
                save_correct_gaussian_ply(xyz, colors, opacities, scales, rotations, output_path)
        
        print(f"\n导出完成! 文件保存在: {output_dir}")
        
    except Exception as e:
        print(f"从Gaussian模型导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = ArgumentParser(description="正确的Gaussian Splatting PLY导出")
    parser.add_argument('--mode', type=str, choices=['test', 'gaussian'], default='test',
                       help='导出模式: test (测试数据) 或 gaussian (从模型)')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    parser.add_argument('--model_dir', type=str, help='Gaussian模型目录 (gaussian模式)')
    parser.add_argument('--smpl_params', type=str, help='SMPL参数文件 (gaussian模式)')
    parser.add_argument('--frames', type=str, default='0,1,2', help='导出帧索引，逗号分隔 (默认: 0,1,2)')
    
    args = parser.parse_args()
    
    frame_indices = [int(x.strip()) for x in args.frames.split(',')]
    
    if args.mode == 'test':
        # 创建测试数据
        print("创建测试Gaussian数据...")
        xyz, colors, opacities, scales, rotations = create_test_gaussian_data(5000)
        
        output_path = args.output
        if not output_path.endswith('.ply'):
            output_path = os.path.join(output_path, 'test_gaussian.ply')
        
        save_correct_gaussian_ply(xyz, colors, opacities, scales, rotations, output_path)
        
    elif args.mode == 'gaussian':
        # 从Gaussian模型导出
        if not args.model_dir or not args.smpl_params:
            print("错误: gaussian模式需要 --model_dir 和 --smpl_params 参数")
            return
        
        export_from_gaussian_model_corrected(
            args.model_dir, args.smpl_params, args.output, frame_indices
        )
    
    print("\n使用建议:")
    print("1. 在Gaussian Splatting查看器中测试显示效果")
    print("2. 如果点仍然太大，可以进一步减小缩放限制")
    print("3. 确保查看器支持标准3DGS格式")


if __name__ == "__main__":
    main()
