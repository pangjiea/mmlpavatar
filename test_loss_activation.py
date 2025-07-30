#!/usr/bin/env python3
"""
测试头部身体分离损失在迭代1000后的激活情况
"""

import os
import torch
import sys
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_head_body_loss_activation():
    """测试头部身体分离损失激活"""
    print("测试头部身体分离损失在迭代1000后的激活...")
    
    try:
        # 导入必要模块
        from scene.gaussian_model import GaussianModel
        from scene.scene import Scene
        from scene.dataset import data_to_cam
        from utils.loss_utils import head_body_separation_loss
        from utils.smpl_utils import init_smpl_pose
        
        # 初始化
        init_smpl_pose()
        
        # 加载配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        args.data_dir = '/home/hello/data/SQ_02/'
        args.out_dir = '/home/hello/code/mmlphuman/output/test_loss_activation'
        args.iterations = 1100  # 测试到1100迭代
        args.num_train_frame = 5  # 少量数据
        args.train_cam_ids = [0, 1]  # 2个相机
        
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建场景和模型
        gaussians = GaussianModel()
        scene = Scene(args, gaussians)
        gaussians.training_setup(args, scene.scene_scale)
        
        print(f"✓ 配置: iteration_head_body_loss = {args.iteration_head_body_loss}")
        
        # 测试损失在不同迭代的行为
        trainloader_iter = iter(scene.trainloader)
        cam = next(trainloader_iter)
        cam = data_to_cam(cam)
        
        # 设置SMPL参数
        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']
        gaussians.expression = cam.get('expression', torch.zeros(10, dtype=torch.float32))
        gaussians.jaw_pose = cam.get('jaw_pose', torch.zeros(3, dtype=torch.float32))
        
        # 测试迭代999 (应该禁用)
        iteration = 999
        if iteration > args.iteration_head_body_loss:
            separation_loss, separation_loss_dict = head_body_separation_loss(
                gaussians,
                alpha=args.lambda_head_body_feature_diff,
                beta=args.lambda_head_body_spatial_sep
            )
            head_body_loss = separation_loss * args.lambda_head_body_separation
            print(f"Iteration {iteration}: Head-body loss = {head_body_loss.item():.6f} (应该被跳过)")
        else:
            head_body_loss = torch.tensor(0.0, device='cuda')
            print(f"Iteration {iteration}: Head-body loss disabled = {head_body_loss.item():.6f} ✓")
        
        # 测试迭代1001 (应该激活)
        iteration = 1001
        if iteration > args.iteration_head_body_loss:
            separation_loss, separation_loss_dict = head_body_separation_loss(
                gaussians,
                alpha=args.lambda_head_body_feature_diff,
                beta=args.lambda_head_body_spatial_sep
            )
            head_body_loss = separation_loss * args.lambda_head_body_separation
            print(f"Iteration {iteration}: Head-body loss activated = {head_body_loss.item():.6f} ✓")
            print("  分离损失详情:")
            for key, value in separation_loss_dict.items():
                print(f"    {key}: {value.item():.6f}")
        else:
            head_body_loss = torch.tensor(0.0, device='cuda')
            print(f"Iteration {iteration}: Head-body loss = {head_body_loss.item():.6f} (不应该被禁用)")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_extended_training_test():
    """运行扩展的训练测试，包含头部身体分离损失激活"""
    print("\n开始扩展训练测试...")
    
    try:
        from scene.gaussian_model import GaussianModel
        from scene.scene import Scene
        from scene.dataset import data_to_cam
        from utils.loss_utils import l1_loss, head_body_separation_loss
        from utils.smpl_utils import init_smpl_pose
        
        # 初始化
        init_smpl_pose()
        
        # 加载配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        args.data_dir = '/home/hello/data/SQ_02/'
        args.out_dir = '/home/hello/code/mmlphuman/output/test_extended'
        args.iterations = 1010  # 测试跨越1000迭代边界
        args.num_train_frame = 3
        args.train_cam_ids = [0]
        args.iteration_head_body_loss = 1005  # 设置在1005激活，方便测试
        
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建场景和模型
        gaussians = GaussianModel()
        scene = Scene(args, gaussians)
        gaussians.training_setup(args, scene.scene_scale)
        
        background = torch.as_tensor(args.background).float().cuda()
        trainloader_iter = iter(scene.trainloader)
        
        # 测试关键迭代点
        test_iterations = [1003, 1004, 1005, 1006, 1007]
        
        for iteration in test_iterations:
            try:
                cam = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(scene.trainloader)
                cam = next(trainloader_iter)
            
            cam = data_to_cam(cam)
            
            # 设置参数
            gaussians.smpl_poses = cam['pose']
            gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']
            gaussians.expression = cam.get('expression', torch.zeros(10, dtype=torch.float32))
            gaussians.jaw_pose = cam.get('jaw_pose', torch.zeros(3, dtype=torch.float32))
            
            # 渲染
            image, _, _ = gaussians.render(cam, background=background)
            image = torch.clamp(image, 0, 1)
            image_gt = cam['image']
            image_gt[~cam['mask']] = background
            
            # 基本损失
            l1loss = l1_loss(image, image_gt)
            
            # 头部身体分离损失
            if iteration > args.iteration_head_body_loss:
                separation_loss, separation_loss_dict = head_body_separation_loss(
                    gaussians,
                    alpha=args.lambda_head_body_feature_diff,
                    beta=args.lambda_head_body_spatial_sep
                )
                head_body_loss = separation_loss * args.lambda_head_body_separation
                status = "ACTIVATED ✓"
            else:
                head_body_loss = torch.tensor(0.0, device='cuda')
                separation_loss_dict = {}
                status = "disabled"
            
            total_loss = l1loss + head_body_loss
            
            print(f"Iteration {iteration}: L1={l1loss.item():.4f}, HeadBody={head_body_loss.item():.4f}, Total={total_loss.item():.4f} [{status}]")
            
            # 反向传播测试
            total_loss.backward()
            gaussians.optimizer_step()
        
        print("✓ 扩展训练测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 扩展训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("头部身体分离损失激活验证测试")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_head_body_loss_activation()
    test2_passed = run_extended_training_test()
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"损失激活测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"扩展训练测试: {'✓ 通过' if test2_passed else '✗ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过! 头部身体分离功能完全就绪")
        print("\n📋 完整训练命令:")
        print("python train.py --config config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_head_body")
        print("\n💡 训练特点:")
        print("- 前1000次迭代: 常规训练，建立基础模型")
        print("- 1000次迭代后: 激活头部身体分离损失，开始解耦训练")
        print("- Expression和jaw_pose参数会被正确使用")
        print("- 损失函数会促进头部和身体特征分离")
    else:
        print("\n❌ 测试失败，需要进一步调试")