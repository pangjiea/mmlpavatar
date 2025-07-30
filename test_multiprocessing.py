#!/usr/bin/env python3
"""
测试恢复多进程后的训练速度
"""

import os
import torch
import sys
import time
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/hello/code/mmlphuman')

def test_multiprocessing():
    """测试多进程训练"""
    print("测试恢复多进程后的训练...")
    
    try:
        from scene.gaussian_model import GaussianModel
        from scene.scene import Scene
        from scene.dataset import data_to_cam
        from utils.loss_utils import l1_loss
        from utils.smpl_utils import init_smpl_pose
        
        # 初始化
        init_smpl_pose()
        
        # 加载配置
        config_path = '/home/hello/code/mmlphuman/config/SQ_02.yaml'
        args = OmegaConf.load(config_path)
        args.data_dir = '/home/hello/data/SQ_02/'
        args.out_dir = '/home/hello/code/mmlphuman/output/test_multiprocessing'
        args.iterations = 20  # 测试20迭代
        args.num_train_frame = 10
        args.train_cam_ids = [0, 1, 2, 3]
        
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建场景和模型
        print("创建场景和模型...")
        gaussians = GaussianModel()
        scene = Scene(args, gaussians)
        gaussians.training_setup(args, scene.scene_scale)
        
        background = torch.as_tensor(args.background).float().cuda()
        trainloader_iter = iter(scene.trainloader)
        
        print(f"✓ 场景创建成功，数据加载器num_workers=8")
        print(f"✓ 训练数据: {len(scene.trainset)} images")
        
        # 测试训练速度
        print("开始速度测试...")
        start_time = time.time()
        
        for iteration in range(1, 11):  # 测试10次迭代
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
            
            # 检查数据
            if iteration == 1:
                print(f"✓ Expression shape: {gaussians.expression.shape}")
                print(f"✓ Jaw pose shape: {gaussians.jaw_pose.shape}")
                print(f"✓ Mask有效像素: {cam['mask'].sum().item()}")
            
            # 渲染
            image, alpha, info = gaussians.render(cam, background=bg)
            image = torch.clamp(image, 0, 1)
            image_gt, mask = cam['image'], cam['mask']
            image_gt[~mask] = bg
            
            # 计算损失
            l1loss = l1_loss(image, image_gt)
            
            # 反向传播
            l1loss.backward()
            gaussians.optimizer_step()
            
            if iteration % 5 == 0:
                elapsed = time.time() - start_time
                speed = iteration / elapsed
                print(f"Iteration {iteration}: {speed:.2f} it/s, L1={l1loss.item():.4f}")
        
        total_time = time.time() - start_time
        avg_speed = 10 / total_time
        
        print(f"\n✓ 多进程训练测试成功!")
        print(f"✓ 平均速度: {avg_speed:.2f} iterations/second")
        print(f"✓ 总用时: {total_time:.2f} seconds")
        
        if avg_speed > 0.5:  # 如果速度合理
            print("🚀 速度恢复正常，可以开始训练!")
            return True
        else:
            print("⚠️ 速度仍然较慢，可能有其他问题")
            return False
        
    except Exception as e:
        print(f"✗ 多进程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("多进程训练速度测试")
    print("=" * 60)
    
    success = test_multiprocessing()
    
    if success:
        print("\n🎉 多进程恢复成功! 现在可以快速训练:")
        print("python train.py --config ./config/SQ_02.yaml --data_dir /home/hello/data/SQ_02/ --out_dir output/sq_02_head_body")
    else:
        print("\n❌ 多进程仍有问题，可能需要进一步调试")