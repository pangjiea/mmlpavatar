import os
import json
import numpy as np
from os import path

def convert_smplx_to_npz(data_dir):
    """将SMPL-X JSON文件转换为NPZ格式"""
    smplx_dir = path.join(data_dir, 'smplx_fitting')
    
    # 获取所有JSON文件
    json_files = sorted([f for f in os.listdir(smplx_dir) if f.endswith('.json')])
    
    if not json_files:
        print("未找到SMPL-X JSON文件")
        return
    
    # 读取第一个文件获取beta参数
    first_file = path.join(smplx_dir, json_files[0])
    with open(first_file, 'r') as f:
        first_data = json.load(f)
    
    betas = np.array(first_data['betas']).astype(np.float32)  # 形状: (1, 300)
    
    # 初始化列表存储所有帧的数据
    global_orient_list = []
    body_pose_list = []
    left_hand_pose_list = []
    right_hand_pose_list = []
    transl_list = []
    expression_list = []
    jaw_pose_list = []
    
    print(f"转换 {len(json_files)} 个文件...")
    
    for json_file in json_files:
        json_path = path.join(smplx_dir, json_file)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 提取参数
        global_orient = np.array(data['global_orient'][0]).astype(np.float32)
        body_pose = np.array(data['body_pose'][0]).astype(np.float32)  
        left_hand_pose = np.array(data['left_hand_pose'][0]).astype(np.float32)
        right_hand_pose = np.array(data['right_hand_pose'][0]).astype(np.float32)
        transl = np.array(data['transl'][0]).astype(np.float32)
        
        # SMPL-X特有的参数
        expression = np.array(data['expression'][0]).astype(np.float32)
        jaw_pose = np.array(data['jaw_pose'][0]).astype(np.float32)
        
        global_orient_list.append(global_orient)
        body_pose_list.append(body_pose)
        left_hand_pose_list.append(left_hand_pose)
        right_hand_pose_list.append(right_hand_pose)
        transl_list.append(transl)
        expression_list.append(expression)
        jaw_pose_list.append(jaw_pose)
    
    # 转换为numpy数组
    smpl_params = {
        'betas': betas,  # (1, 300)
        'global_orient': np.array(global_orient_list),  # (N, 3)
        'body_pose': np.array(body_pose_list),  # (N, 63)
        'left_hand_pose': np.array(left_hand_pose_list),  # (N, 45)
        'right_hand_pose': np.array(right_hand_pose_list),  # (N, 45)
        'transl': np.array(transl_list),  # (N, 3)
        'expression': np.array(expression_list),  # (N, 50)
        'jaw_pose': np.array(jaw_pose_list),  # (N, 3)
    }
    
    # 保存为NPZ文件
    output_path = path.join(data_dir, 'smpl_params.npz')
    np.savez(output_path, **smpl_params)
    
    print(f"已保存到: {output_path}")
    print(f"形状信息:")
    for key, value in smpl_params.items():
        print(f"  {key}: {value.shape}")

def merge_json_to_npz(data_dir):
    """将指定目录中的所有 JSON 文件合并为一个 smpl_params.npz 文件"""
    smplx_dir = path.join(data_dir, 'smplx_fitting')
    
    # 获取所有JSON文件
    json_files = sorted([f for f in os.listdir(smplx_dir) if f.endswith('.json')])
    
    if not json_files:
        print("未找到SMPL-X JSON文件")
        return
    
    # 初始化字典存储所有帧的数据
    smpl_params = {}
    
    print(f"转换 {len(json_files)} 个文件...")
    
    for json_file in json_files:
        json_path = path.join(smplx_dir, json_file)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            # 将body_pose重命名为pose
            if key == 'body_pose':
                key = 'pose'
            
            if key not in smpl_params:
                smpl_params[key] = []
            smpl_params[key].append(np.array(value).astype(np.float32))
    
    # 转换为numpy数组并压缩多余的维度
    for key in smpl_params:
        smpl_params[key] = np.array(smpl_params[key]).squeeze(axis=1)
    
    output_path = path.join(data_dir, 'smpl_params.npz')
    np.savez(output_path, **smpl_params)
    
    print(f"已保存合并文件到: {output_path}")
    print(f"形状信息:")
    for key, value in smpl_params.items():
        print(f"  {key}: {value.shape}")

if __name__ == "__main__":
    data_dir = "/home/hello/data/SQ_02"
    merge_json_to_npz(data_dir)