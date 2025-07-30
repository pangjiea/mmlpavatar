#!/usr/bin/env python3
"""
测试修复后的训练是否正常
"""

import os
import torch
import sys
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_fixed_training():
    """测试修复后的训练"""
    print("测试修复后的训练...")
    
    try:
        # 导入必要模块
        from scene.gaussian_model import GaussianModel
        from scene.scene import Scene
        from scene.dataset import data_to_cam
        from utils.loss_utils import l1_loss
        from utils.smpl_utils import init_smpl_pose
        from utils.image_utils import crop_image
        
        print("✓ 模块导入成功")
        
        # 初始化
        init_smpl_pose()
        
        # 加载配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        args.data_dir = '/home/hello/data/SQ_02/'
        args.out_dir = '/home/hello/code/mmlphuman/output/test_fixed'
        args.iterations = 10  # 只测试少量迭代
        args.num_train_frame = 3
        args.train_cam_ids = [0]
        
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建场景和模型
        gaussians = GaussianModel()
        scene = Scene(args, gaussians)
        gaussians.training_setup(args, scene.scene_scale)
        
        background = torch.as_tensor(args.background).float().cuda()
        trainloader_iter = iter(scene.trainloader)
        
        print("✓ 场景和模型创建成功")
        
        # 测试训练循环
        for iteration in range(1, 4):
            try:
                cam = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(scene.trainloader)
                cam = next(trainloader_iter)
            
            cam = data_to_cam(cam)
            bg = background
            
            # 设置参数
            gaussians.smpl_poses = cam['pose']
            gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']
            gaussians.expression = cam.get('expression', torch.zeros(10, dtype=torch.float32))
            gaussians.jaw_pose = cam.get('jaw_pose', torch.zeros(3, dtype=torch.float32))
            
            # 渲染
            image, alpha, info = gaussians.render(cam, background=bg)
            image = torch.clamp(image, 0, 1)
            image_gt, mask = cam['image'], cam['mask']
            image_gt[~mask] = bg
            
            print(f"Iteration {iteration}: mask有效像素数 = {mask.sum().item()}")
            
            # 测试图像裁剪
            try:
                random_patch_flag = False
                image_crop, image_gt_crop = crop_image(bg, mask, 512, random_patch_flag, image.permute(2,0,1), image_gt.permute(2,0,1))
                print(f"✓ Iteration {iteration}: 图像裁剪成功，尺寸 = {image_crop.shape}")
            except Exception as e:
                print(f"✗ Iteration {iteration}: 图像裁剪失败 - {e}")
                return False
            
            # 计算损失
            l1loss = l1_loss(image, image_gt)
            
            # 反向传播
            l1loss.backward()
            gaussians.optimizer_step()
            
            print(f"✓ Iteration {iteration}: L1 loss = {l1loss.item():.6f}")
        
        print("✓ 训练循环测试成功完成")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("测试修复后的训练功能")
    print("=" * 50)
    
    success = test_fixed_training()
    
    if success:
        print("\n🎉 修复成功! 可以开始正常训练")
        print("运行命令:")
        print("python train.py --config ./config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_head_body")
    else:
        print("\n❌ 仍有问题，需要进一步调试")