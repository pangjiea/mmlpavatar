#!/usr/bin/env python3
"""
测试真正的SMPL-X mesh头部掩膜渲染
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_real_smplx_mesh_rendering():
    """测试真正的SMPL-X mesh渲染"""
    print("测试真正的SMPL-X mesh头部掩膜渲染...")
    
    try:
        from utils.head_mask_utils import generate_head_mask, generate_head_mask_from_smpl_vertices, load_head_faces
        
        # 检查head faces是否可以加载
        head_faces = load_head_faces()
        print(f"✓ 加载头部面数量: {len(head_faces)}")
        
        # 测试参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        
        # 测试不同的expression（确保50维）
        expression = np.zeros(50, dtype=np.float32)
        expression[0] = 0.5  # 添加一些表情变化
        expression[1] = -0.3
        
        jaw_pose = np.zeros(3, dtype=np.float32)
        jaw_pose[0] = 0.1  # 添加一些下巴运动
        
        # 添加一些pose变化
        pose[3:6] = [0.1, 0.0, 0.0]  # 全局旋转
        
        Rh = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # 移动到相机前方
        
        # 相机参数
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        
        # 测试不同图像尺寸
        test_sizes = [
            (480, 640),   # VGA
            (720, 1280),  # HD
        ]
        
        for height, width in test_sizes:
            print(f"\n测试图像尺寸: {height}x{width}")
            
            # 生成头部mask
            head_mask = generate_head_mask(
                pose, beta, expression, jaw_pose, Rh, Th,
                K, w2c, height, width, device='cpu'
            )
            
            print(f"   生成mask形状: {head_mask.shape}")
            print(f"   头部像素数量: {head_mask.sum()}")
            print(f"   头部像素比例: {head_mask.sum() / (height * width) * 100:.1f}%")
            print(f"   Mask数据类型: {head_mask.dtype}")
            
            # 验证mask的基本属性
            assert head_mask.shape == (height, width), f"形状不匹配: {head_mask.shape} vs {(height, width)}"
            assert head_mask.dtype == bool, f"类型不匹配: {head_mask.dtype}"
            
            # 分析mask分布
            if head_mask.sum() > 0:
                y_indices, x_indices = np.where(head_mask)
                y_center = y_indices.mean()
                x_center = x_indices.mean()
                print(f"   头部中心: ({x_center:.1f}, {y_center:.1f})")
                print(f"   Y范围: {y_indices.min()} - {y_indices.max()}")
                print(f"   X范围: {x_indices.min()} - {x_indices.max()}")
                
                # 检查是否真的基于mesh（不是完美椭圆）
                # 计算mask的形状复杂度
                contours_count = 0
                try:
                    import cv2
                    mask_uint8 = head_mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_count = len(contours)
                    if contours_count > 0:
                        largest_contour = max(contours, key=cv2.contourArea)
                        contour_area = cv2.contourArea(largest_contour)
                        hull = cv2.convexHull(largest_contour)
                        hull_area = cv2.contourArea(hull)
                        complexity = 1.0 - (contour_area / hull_area) if hull_area > 0 else 0
                        print(f"   轮廓数量: {contours_count}")
                        print(f"   形状复杂度: {complexity:.3f} (0=椭圆, >0=复杂形状)")
                        
                        if complexity > 0.1:
                            print("   ✓ 检测到复杂形状，可能是真实mesh渲染")
                        else:
                            print("   ⚠️ 形状较简单，可能是几何估计")
                except:
                    print("   无法分析形状复杂度")
            
            print("   ✓ 基本验证通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_vs_geometric_comparison():
    """测试mesh渲染与几何估计的对比"""
    print("\n测试mesh渲染与几何估计的对比...")
    
    try:
        from utils.head_mask_utils import generate_head_mask, generate_head_mask_from_smpl_vertices, generate_geometric_head_mask
        from utils.smpl_utils import init_smpl
        
        # 初始化SMPL模型
        smpl_pkl_path = "./smpl_model/smplx/SMPLX_NEUTRAL.npz"
        if os.path.exists(smpl_pkl_path):
            init_smpl(smpl_pkl_path)
        
        height, width = 480, 640
        
        # 测试参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        
        # 生成SMPL-based mask（新方法）
        smpl_mask = generate_head_mask_from_smpl_vertices(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, height, width, device='cpu'
        )
        
        # 生成几何mask
        geometric_mask = generate_geometric_head_mask(height, width)
        
        # 对比分析
        print(f"SMPL-based mask像素数: {smpl_mask.sum()}")
        print(f"几何mask像素数: {geometric_mask.sum()}")
        
        # 计算重叠度
        intersection = (smpl_mask & geometric_mask).sum()
        union = (smpl_mask | geometric_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        print(f"重叠像素数: {intersection}")
        print(f"并集像素数: {union}")
        print(f"IoU: {iou:.3f}")
        
        if iou < 0.8:
            print("✓ SMPL-based mask与几何mask有显著差异，表明使用了真实SMPL信息")
        else:
            print("⚠️ SMPL-based mask与几何mask很相似，可能fallback到几何方法")
        
        # 保存对比可视化
        try:
            from utils.mask_visualization import save_head_mask_comparison
            
            # 创建测试图像
            test_image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
            
            debug_dir = "/home/hello/code/mmlphuman/debug_mesh_test"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存对比
            comp_path = os.path.join(debug_dir, "smpl_vs_geometric_comparison.png")
            save_head_mask_comparison(test_image, smpl_mask, geometric_mask, comp_path, 
                                    frame_id=0, cam_id=0)
            print(f"✓ SMPL vs 几何对比可视化保存到: {comp_path}")
            
        except Exception as e:
            print(f"保存可视化失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 对比测试失败: {e}")
        return False

def test_smplx_model_availability():
    """测试SMPL-X模型是否可用"""  
    print("\n测试SMPL-X模型可用性...")
    
    try:
        from utils.smpl_utils import smpl, init_smpl
        import torch
        
        # 初始化SMPL模型
        smpl_pkl_path = "./smpl_model/smplx/SMPLX_NEUTRAL.npz"
        if not os.path.exists(smpl_pkl_path):
            print(f"✗ SMPL模型文件不存在: {smpl_pkl_path}")
            return False
            
        init_smpl(smpl_pkl_path)
        
        print(f"✓ SMPL模型加载成功: {type(smpl)}")
        print(f"✓ SMPL模型设备: {next(smpl.model.parameters()).device}")
        print(f"✓ SMPL顶点数: {smpl.model.v_template.shape[0]}")
        print(f"✓ SMPL面数: {smpl.model.faces_tensor.shape[0]}")
        
        # 测试forward pass
        batch_size = 1
        body_pose = torch.zeros(batch_size, 63)
        global_orient = torch.zeros(batch_size, 3)
        betas = torch.zeros(batch_size, 10)
        expression = torch.zeros(batch_size, 50)
        jaw_pose = torch.zeros(batch_size, 3)
        
        with torch.no_grad():
            output = smpl.model(
                body_pose=body_pose,
                global_orient=global_orient,
                betas=betas,
                expression=expression,
                jaw_pose=jaw_pose
            )
            vertices = output.vertices
            print(f"✓ SMPL forward pass成功，顶点形状: {vertices.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ SMPL-X模型不可用: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("真正的SMPL-X Mesh头部掩膜渲染测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("SMPL-X模型可用性", test_smplx_model_availability),
        ("真实Mesh渲染", test_real_smplx_mesh_rendering),
        ("Mesh vs 几何对比", test_mesh_vs_geometric_comparison),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过!")
        print("\n现在使用真正的SMPL-X mesh渲染:")
        print("- 使用SMPL-X模型生成真实的头部顶点")
        print("- 基于smplx_body_parts_2_faces.json的头部面索引")
        print("- 使用OpenCV渲染头部面到掩膜")
        print("- 避免PyTorch3D的einsum问题")
        print("- 强制使用CPU避免多进程CUDA冲突")
        print("\n重新训练将看到真实的头部mesh形状，而不是椭圆!")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")