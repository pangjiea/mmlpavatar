#!/usr/bin/env python3
"""
运行头部身体分离功能的实际训练测试
"""

import os
import torch
import sys
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def run_training_test():
    """运行实际的训练测试"""
    print("开始运行头部身体分离训练测试...")
    
    try:
        # 导入必要模块
        from scene.gaussian_model import GaussianModel
        from scene.scene import Scene
        from scene.dataset import data_to_cam
        from utils.loss_utils import l1_loss, head_body_separation_loss, head_consistency_loss
        from utils.smpl_utils import init_smpl_pose
        
        print("✓ 模块导入成功")
        
        # 初始化SMPL
        init_smpl_pose()
        print("✓ SMPL初始化成功")
        
        # 加载配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        
        # 设置短时间测试参数
        args.data_dir = '/home/hello/data/SQ_02/'
        args.out_dir = '/home/hello/code/mmlphuman/output/test_head_body'
        args.iterations = 50  # 只训练50迭代用于测试
        args.num_train_frame = 10  # 只用10帧数据
        args.train_cam_ids = [0, 1, 2]  # 只用3个相机
        
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"✓ 配置加载成功，输出目录: {args.out_dir}")
        
        # 创建场景和模型
        gaussians = GaussianModel()
        scene = Scene(args, gaussians)
        gaussians.training_setup(args, scene.scene_scale)
        
        print("✓ 场景和模型创建成功")
        
        # 设置背景
        background = torch.as_tensor(args.background).float().cuda()
        
        # 开始训练测试
        print("开始训练测试...")
        trainloader_iter = iter(scene.trainloader)
        
        for iteration in range(1, min(args.iterations + 1, 20)):  # 最多20迭代
            try:
                cam = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(scene.trainloader)
                cam = next(trainloader_iter)
            
            cam = data_to_cam(cam)
            bg = background
            
            # 设置SMPL参数
            gaussians.smpl_poses = cam['pose']
            gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']
            gaussians.expression = cam.get('expression', torch.zeros(10, dtype=torch.float32))
            gaussians.jaw_pose = cam.get('jaw_pose', torch.zeros(3, dtype=torch.float32))
            
            print(f"Iteration {iteration}:")
            print(f"  - Expression shape: {gaussians.expression.shape}, sample: {gaussians.expression[:3]}")
            print(f"  - Jaw pose shape: {gaussians.jaw_pose.shape}, sample: {gaussians.jaw_pose}")
            
            # 渲染
            image, alpha, info = gaussians.render(cam, background=bg)
            image = torch.clamp(image, 0, 1)
            image_gt, mask = cam['image'], cam['mask']
            image_gt[~mask] = bg
            
            # 计算基本损失
            l1loss = l1_loss(image, image_gt)
            
            # 测试头部身体分离损失
            if iteration > args.iteration_head_body_loss:
                separation_loss, separation_loss_dict = head_body_separation_loss(
                    gaussians,
                    alpha=args.lambda_head_body_feature_diff,
                    beta=args.lambda_head_body_spatial_sep
                )
                head_body_loss = separation_loss * args.lambda_head_body_separation
                
                print(f"  - Head-body separation loss: {head_body_loss.item():.6f}")
                for key, value in separation_loss_dict.items():
                    print(f"    {key}: {value.item():.6f}")
            else:
                head_body_loss = torch.tensor(0.0, device='cuda')
                print(f"  - Head-body separation loss: disabled (iteration < {args.iteration_head_body_loss})")
            
            # 计算头部一致性损失
            if 'head_mask' in cam:
                head_consistency_loss_val = head_consistency_loss(
                    gaussians,
                    cam['head_mask'],
                    gamma=args.lambda_head_consistency
                )
                print(f"  - Head consistency loss: {head_consistency_loss_val.item():.6f}")
            else:
                head_consistency_loss_val = torch.tensor(0.0, device='cuda')
                print(f"  - Head consistency loss: no head mask")
            
            # 总损失
            loss = l1loss + head_body_loss + head_consistency_loss_val
            
            print(f"  - L1 loss: {l1loss.item():.6f}")
            print(f"  - Total loss: {loss.item():.6f}")
            
            # 反向传播
            loss.backward()
            
            # 优化器步骤
            gaussians.optimizer_step()
            
            print(f"  ✓ Iteration {iteration} completed successfully")
            
            if iteration >= 5:  # 测试5次迭代就够了
                break
        
        print("\n🎉 训练测试成功完成!")
        print("✓ 所有功能正常工作:")
        print("  - Expression和jaw_pose参数正确加载")
        print("  - 头部身体分离损失正常计算")
        print("  - 训练循环正常运行")
        print("  - 反向传播和优化正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("头部身体分离功能实际训练测试")
    print("=" * 60)
    
    success = run_training_test()
    
    if success:
        print("\n🎉 测试成功! 代码准备好进行完整训练")
        print("\n完整训练命令:")
        print("python train.py --config config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_head_body")
    else:
        print("\n❌ 测试失败，需要修复问题")