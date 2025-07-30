#!/usr/bin/env python3
"""
测试数据加载器是否还有NoneType错误
"""

import sys
sys.path.append('/home/hello/code/mmlphuman')

from scene.dataset import AVRexDataset
from utils.config_utils import load_config

# 加载配置
config = load_config('/home/hello/code/mmlphuman/config/SQ_02.yaml')

# 创建数据集
try:
    dataset = AVRexDataset(
        '/home/hello/data/SQ_02/',
        'train',
        config['train']['white_background'],
        config['train']['head_body_separation']['lambda_head_body_separation'],
        # 模拟训练设置
        train=True,
        hold_id=0,
        split_seed=42,
        split_ratio=0.9
    )
    
    print(f"✓ 数据集创建成功: {len(dataset)} 个样本")
    
    # 测试前几个样本
    for i in range(min(3, len(dataset))):
        print(f"\n测试样本 {i}...")
        try:
            sample = dataset[i]
            
            # 检查head_mask
            head_mask = sample['head_mask']
            print(f"  head_mask类型: {type(head_mask)}")
            print(f"  head_mask形状: {head_mask.shape}")
            print(f"  head_mask像素数: {head_mask.sum()}")
            print(f"  ✅ 样本 {i} 成功")
            
        except Exception as e:
            print(f"  ❌ 样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
            break
    else:
        print(f"\n🎉 所有测试样本加载成功！数据加载器已修复")
        
except Exception as e:
    print(f"❌ 数据集创建失败: {e}")
    import traceback
    traceback.print_exc()