#!/usr/bin/env python3
"""
训练过程中的头部mask监控工具
"""

import os
import numpy as np
import torch
from utils.mask_visualization import save_head_mask_visualization, save_head_mask_comparison

class HeadMaskMonitor:
    """头部mask训练监控器"""
    
    def __init__(self, output_dir, max_samples=20, save_interval=1000):
        """
        Args:
            output_dir: 输出目录
            max_samples: 最大保存样本数
            save_interval: 保存间隔（迭代数）
        """
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.save_interval = save_interval
        self.sample_count = 0
        self.last_save_iter = 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
        
    def should_save(self, iteration):
        """判断是否应该保存"""
        return (iteration - self.last_save_iter >= self.save_interval and 
                self.sample_count < self.max_samples)
    
    def save_training_sample(self, data_dict, gaussians, iteration):
        """
        保存训练样本的头部mask可视化
        
        Args:
            data_dict: 训练数据字典
            gaussians: GaussianModel实例
            iteration: 当前迭代数
        """
        
        if not self.should_save(iteration):
            return
            
        try:
            # 提取数据
            image = data_dict['image']  # [3, H, W] tensor
            head_mask = data_dict.get('head_mask', None)  # [H, W] tensor
            
            if head_mask is None:
                print("No head mask found in data_dict")
                return
                
            # 转换为numpy
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.cpu().numpy()
                
            # 确保图像范围正确
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # 保存路径
            frame_id = data_dict.get('frame_id', self.sample_count)
            cam_id = data_dict.get('cam_id', 0)
            
            mask_path = os.path.join(self.output_dir, "masks", 
                                   f"iter_{iteration:06d}_sample_{self.sample_count:03d}_frame_{frame_id:06d}.png")
            
            # 保存mask可视化
            save_head_mask_visualization(
                image, head_mask, mask_path,
                title=f'Training Sample - Iter {iteration}, Frame {frame_id}'
            )
            
            # 如果有身体信息，保存对比
            if hasattr(gaussians, 'head_gs_mask') and gaussians.head_gs_mask is not None:
                comp_path = os.path.join(self.output_dir, "comparisons",
                                       f"iter_{iteration:06d}_sample_{self.sample_count:03d}_headbody.png")
                
                # 创建简单的身体mask（非头部区域）
                body_mask = ~head_mask
                save_head_mask_comparison(image, head_mask, body_mask, comp_path, frame_id, cam_id)
            
            self.sample_count += 1
            self.last_save_iter = iteration
            
            print(f"Saved head mask visualization: {mask_path}")
            
        except Exception as e:
            print(f"Failed to save training head mask sample: {e}")
    
    def save_gaussian_separation(self, gaussians, iteration, image_shape=(480, 640)):
        """
        保存Gaussian点的头部-身体分离可视化
        
        Args:
            gaussians: GaussianModel实例
            iteration: 当前迭代数
            image_shape: 图像尺寸 (H, W)
        """
        
        if not self.should_save(iteration):
            return
            
        try:
            if (hasattr(gaussians, 'head_gs_mask') and gaussians.head_gs_mask is not None and
                hasattr(gaussians, 'body_gs_mask') and gaussians.body_gs_mask is not None):
                
                # 创建可视化图像
                vis_image = np.zeros((*image_shape, 3), dtype=np.uint8)
                
                # 获取Gaussian点的位置（如果可用）
                if hasattr(gaussians, '_xyz') and gaussians._xyz is not None:
                    xyz = gaussians._xyz.detach().cpu().numpy()  # [N, 3]
                    
                    # 简单的投影到图像平面（假设正交投影）
                    x_coords = ((xyz[:, 0] + 2) / 4 * image_shape[1]).astype(int)
                    y_coords = ((xyz[:, 1] + 2) / 4 * image_shape[0]).astype(int)
                    
                    # 过滤在图像范围内的点
                    valid_mask = ((x_coords >= 0) & (x_coords < image_shape[1]) & 
                                 (y_coords >= 0) & (y_coords < image_shape[0]))
                    
                    x_coords = x_coords[valid_mask]
                    y_coords = y_coords[valid_mask]
                    head_mask = gaussians.head_gs_mask[valid_mask].cpu().numpy()
                    
                    # 绘制点
                    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                        color = [255, 0, 0] if head_mask[i] else [0, 0, 255]  # 红色=头部，蓝色=身体
                        vis_image[y-1:y+2, x-1:x+2] = color
                
                # 保存
                save_path = os.path.join(self.output_dir, "comparisons",
                                       f"iter_{iteration:06d}_gaussian_separation.png")
                
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(vis_image)
                plt.title(f'Gaussian Head-Body Separation - Iteration {iteration}\nRed=Head, Blue=Body')
                plt.axis('off')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved Gaussian separation visualization: {save_path}")
                
        except Exception as e:
            print(f"Failed to save Gaussian separation visualization: {e}")

# 全局监控器实例
_global_head_mask_monitor = None

def get_head_mask_monitor(output_dir=None):
    """获取全局头部mask监控器"""
    global _global_head_mask_monitor
    
    if _global_head_mask_monitor is None and output_dir is not None:
        _global_head_mask_monitor = HeadMaskMonitor(output_dir)
    
    return _global_head_mask_monitor

def save_training_head_mask_sample(data_dict, gaussians, iteration, output_dir=None):
    """便捷函数：保存训练时的头部mask样本"""
    if output_dir is None:
        output_dir = "/home/hello/code/mmlphuman/debug_training_masks"
    
    monitor = get_head_mask_monitor(output_dir)
    if monitor:
        monitor.save_training_sample(data_dict, gaussians, iteration)

if __name__ == "__main__":
    # 测试代码
    print("Testing head mask monitor...")
    
    # 创建测试数据
    test_image = torch.randn(3, 480, 640)  # [C, H, W]
    test_head_mask = torch.zeros(480, 640, dtype=torch.bool)
    test_head_mask[50:200, 250:390] = True  # 头部区域
    
    test_data = {
        'image': test_image,
        'head_mask': test_head_mask,
        'frame_id': 123,
        'cam_id': 1
    }
    
    # 创建监控器
    monitor = HeadMaskMonitor("test_output", max_samples=5, save_interval=100)
    
    # 模拟保存
    class MockGaussians:
        def __init__(self):
            self.head_gs_mask = None
            self.body_gs_mask = None
    
    mock_gaussians = MockGaussians()
    monitor.save_training_sample(test_data, mock_gaussians, 1000)
    
    print("✅ Head mask monitor test completed!")