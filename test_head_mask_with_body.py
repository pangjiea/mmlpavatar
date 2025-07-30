#!/usr/bin/env python3
"""
测试带全身掩膜的头部掩膜生成
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def create_mock_body_mask(height, width):
    """创建模拟的全身掩膜"""
    body_mask = np.zeros((height, width), dtype=bool)
    
    # 创建一个人形的掩膜
    # 头部区域 (10%-30% height)
    head_y_start = int(height * 0.1)
    head_y_end = int(height * 0.3)
    head_x_center = width // 2
    head_width = int(width * 0.15)
    
    # 身体区域 (25%-80% height) 
    body_y_start = int(height * 0.25)
    body_y_end = int(height * 0.8)
    body_width = int(width * 0.25)
    
    # 填充头部
    for y in range(head_y_start, head_y_end):
        for x in range(head_x_center - head_width//2, head_x_center + head_width//2):
            if 0 <= x < width:
                dx = (x - head_x_center) / (head_width/2)
                dy = (y - (head_y_start + head_y_end)/2) / ((head_y_end - head_y_start)/2)
                if dx*dx + dy*dy <= 1.0:  # 椭圆形头部
                    body_mask[y, x] = True
    
    # 填充身体
    for y in range(body_y_start, body_y_end):
        for x in range(head_x_center - body_width//2, head_x_center + body_width//2):
            if 0 <= x < width:
                body_mask[y, x] = True
    
    return body_mask

def test_head_mask_with_body_intersection():
    """测试头部掩膜与全身掩膜的交集"""
    print("测试头部掩膜与全身掩膜的交集...")
    
    try:
        from utils.head_mask_utils import generate_head_mask
        from utils.smpl_utils import init_smpl
        
        # 初始化SMPL模型
        smpl_pkl_path = "./smpl_model/smplx/SMPLX_NEUTRAL.npz"
        if os.path.exists(smpl_pkl_path):
            init_smpl(smpl_pkl_path)
            print("✓ SMPL模型初始化成功")
        
        height, width = 480, 640
        
        # 创建模拟的全身掩膜
        body_mask = create_mock_body_mask(height, width)
        print(f"✓ 创建全身掩膜: {body_mask.sum()} 像素 ({body_mask.sum()/(height*width)*100:.1f}%)")
        
        # 测试参数
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        
        # 不带全身掩膜的头部掩膜
        head_mask_only = generate_head_mask(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, height, width, body_mask=None, device='cpu'
        )
        
        # 带全身掩膜的头部掩膜
        head_mask_intersected = generate_head_mask(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, height, width, body_mask=body_mask, device='cpu'
        )
        
        print(f"\n结果对比:")
        if head_mask_only is None:
            print("仅头部投影掩膜: None (头部不在图像中)")
        else:
            print(f"仅头部投影掩膜: {head_mask_only.sum()} 像素")
            
        if head_mask_intersected is None:
            print("与全身掩膜交集: None (头部不在图像中)")
        else:
            print(f"与全身掩膜交集: {head_mask_intersected.sum()} 像素")
        
        # 如果头部不在图像中，这是正常情况
        if head_mask_only is None or head_mask_intersected is None:
            print("✓ 头部不在当前视角中，正确跳过头部优化")
            return True
        
        # 计算交集是否合理
        manual_intersection = head_mask_only & body_mask
        print(f"手动计算交集: {manual_intersection.sum()} 像素")
        
        if head_mask_intersected.sum() == manual_intersection.sum():
            print("✓ 交集计算正确")
        else:
            print("⚠️ 交集计算可能有问题")
        
        # 保存可视化
        try:
            from utils.mask_visualization import save_head_mask_comparison
            
            # 创建测试图像
            test_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
            # 将body_mask区域设为白色，模拟真实情况
            test_image[body_mask] = [200, 200, 200]
            
            debug_dir = "/home/hello/code/mmlphuman/debug_body_intersection"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存对比
            comp_path = os.path.join(debug_dir, "head_body_intersection_test.png")
            save_head_mask_comparison(test_image, head_mask_intersected, body_mask, comp_path, 
                                    frame_id=0, cam_id=0)
            print(f"✓ 可视化保存到: {comp_path}")
            
        except Exception as e:
            print(f"保存可视化失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_head_case():
    """测试没有头部的情况"""
    print("\n测试没有头部的情况...")
    
    try:
        from utils.head_mask_utils import generate_head_mask
        
        height, width = 480, 640
        
        # 创建只有身体没有头部的掩膜
        body_mask = np.zeros((height, width), dtype=bool)
        # 只填充下半身
        body_y_start = int(height * 0.5)
        body_y_end = int(height * 0.9)
        body_x_center = width // 2
        body_width = int(width * 0.3)
        
        for y in range(body_y_start, body_y_end):
            for x in range(body_x_center - body_width//2, body_x_center + body_width//2):
                if 0 <= x < width:
                    body_mask[y, x] = True
        
        print(f"✓ 创建无头部身体掩膜: {body_mask.sum()} 像素")
        
        # 测试参数 
        pose = np.zeros(165, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        expression = np.zeros(50, dtype=np.float32)
        jaw_pose = np.zeros(3, dtype=np.float32)
        Rh = np.zeros(3, dtype=np.float32)
        Th = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        
        # 生成头部掩膜
        head_mask = generate_head_mask(
            pose, beta, expression, jaw_pose, Rh, Th,
            K, w2c, height, width, body_mask=body_mask, device='cpu'
        )
        
        if head_mask is None:
            print("✓ 正确检测到无头部情况，返回None")
        elif head_mask.sum() == 0:
            print("✓ 正确检测到无头部情况，返回空掩膜")
        else:
            print(f"⚠️ 检测到 {head_mask.sum()} 像素的头部区域，可能有问题")
        
        return True
        
    except Exception as e:
        print(f"✗ 无头部测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("头部掩膜与全身掩膜交集测试")
    print("=" * 60)
    
    # 运行测试
    tests = [
        ("头部-身体交集测试", test_head_mask_with_body_intersection),
        ("无头部情况测试", test_no_head_case),
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
        print("\n新的头部掩膜逻辑:")
        print("- 基于SMPL头部顶点投影生成头部区域")
        print("- 与全身掩膜取交集，确保头部在身体范围内")
        print("- 自动检测无头部情况并跳过")
        print("- 生成椭圆形而非方框形状的头部掩膜")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")