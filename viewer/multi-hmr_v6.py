import threading
import time
import queue
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
import numpy as np
from PIL import Image, ImageOps
import os
from argparse import ArgumentParser
# from model_temporal import *
from model_rope_transformer import *
import roma
from pathlib import Path
import cv2
import pickle
import random
import socket
import struct
from rot6d import *
def get_gaussian_kernel_1d(kernel_size, sigma, device):
    x = torch.arange(kernel_size).float() - (kernel_size // 2)
    g = torch.exp(-((x**2) / (2 * sigma**2)))
    g /= g.sum()

    kernel_weight = g.view(1, 1, -1).to(device)

    return kernel_weight

def gaussian_filter_1d(data, kernel_size=3, sigma=1.0, weight=None):
    kernel_weight = (
        get_gaussian_kernel_1d(kernel_size, sigma, data.device)
        if weight is None
        else weight
    )
    data = F.pad(data, (kernel_size // 2, kernel_size // 2), mode="replicate")
    return F.conv1d(data, kernel_weight)
def gaussian_filter_1d(data, kernel_size=3, sigma=1.0, weight=None):
    kernel_weight = (
        get_gaussian_kernel_1d(kernel_size, sigma, data.device)
        if weight is None
        else weight
    )
    data = F.pad(data, (kernel_size // 2, kernel_size // 2), mode="replicate")
    return F.conv1d(data, kernel_weight)
def smplx_gs_smooth(poses, betas, transl, fps=30):
    poses = axis_angle_to_rotation_6d(poses)
    N, J, _ = poses.shape
    poses = (
        gaussian_filter_1d(
            poses.view(N, 1, -1).permute(2, 1, 0),
            kernel_size=11,
            sigma=1 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(N, J, -1)
    )
    betas = (
        gaussian_filter_1d(
            betas.view(-1, 1, 10).permute(2, 1, 0),
            kernel_size=11,
            sigma=5.0 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(-1, 10)
    )
    transl= (
        gaussian_filter_1d(
            transl.view(N, 1, -1).permute(2, 1, 0),
            kernel_size=9,
            sigma=1.0 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(N, -1)
    )

    poses = rotation_6d_to_axis_angle(poses)
    return poses, betas, transl

def slerp_batch(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-7):
    """
    对两个同形的四元数张量做批量 SLERP。
    q0, q1: Tensor[..., J, 4]，规范化四元数
    t:   Tensor[...] or float，插值比例，广播到 q0 的前 N-1 维
    返回: Tensor[..., J, 4]
    """
    # 确保单位四元数
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(eps)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)

    # 计算点积，cosθ
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)  # Tensor[..., J, 1]
    # 如果点积为负，取相反四元数以保持最短弧
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # 计算插值角度 θ = arccos(dot)
    theta = torch.acos(dot.clamp(-1+eps, 1-eps))
    sin_theta = torch.sin(theta)

    # 当 sinθ 很小时，退化为线性插值
    small = sin_theta < eps
    # SLERP: (sin((1−t)θ) / sinθ) * q0 + (sin(tθ) / sinθ) * q1
    t_shape = list(dot.shape)
    # 扩展 t 以匹配 q0 的形状 [..., J, 1]
    t_exp = t.view(*t_shape[:-2], 1, 1).expand_as(dot)
    factor0 = torch.sin((1 - t_exp) * theta) / sin_theta
    factor1 = torch.sin(t_exp * theta) / sin_theta

    # 对小角度用 LERP
    out = factor0 * q0 + factor1 * q1
    out = torch.where(small, (1 - t_exp) * q0 + t_exp * q1, out)

    # 归一化输出
    return out / out.norm(dim=-1, keepdim=True).clamp_min(eps)


def axisangle_to_quat_torch(axisangle: torch.Tensor):
    """axisangle: Tensor[..., J, 3] → Tensor[..., J, 4]"""
    angle = axisangle.norm(dim=-1, keepdim=True)
    axis = axisangle / angle.clamp_min(1e-7)
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)

def quat_to_axisangle_torch(quat: torch.Tensor):
    """quat: Tensor[..., J, 4] → Tensor[..., J, 3]"""
    w, xyz = quat[..., :1], quat[..., 1:]
    angle = 2 * torch.acos(w.clamp(-1, 1))
    axis = xyz / torch.sin(angle * 0.5).clamp_min(1e-7)
    return axis * angle

def smooth_translation(t_prev_smooth: torch.Tensor,
                       t_curr_obs: torch.Tensor,
                       a: float = 0.8,
                       b: float = 0.2) -> torch.Tensor:
    """
    对全局平移 (3,) 向量做简单加权平滑。
    
    Args:
        t_prev_smooth: Tensor of shape (3,), 上一帧平滑后的平移向量
        t_curr_obs:     Tensor of shape (3,), 当前帧观测的平移向量
        a:              float, 历史权重
        b:              float, 当前权重  (建议 a + b = 1)
        
    Returns:
        Tensor of shape (3,), 平滑后的全局平移向量
    """
    # 直接做加权求和
    return a * t_prev_smooth + b * t_curr_obs

def smooth_shape_linear(beta_prev_smooth: torch.Tensor,
                        beta_curr_obs: torch.Tensor,
                        a: float = 0.8,
                        b: float = 0.2) -> torch.Tensor:
    """
    对 SMPL-X shape 参数做简单线性加权平滑。

    Args:
        beta_prev_smooth: Tensor of shape (n,), 上一帧平滑后的 beta
        beta_curr_obs:    Tensor of shape (n,), 当前帧观测的 beta
        a:                float, 历史权重
        b:                float, 当前权重（建议 a + b = 1）
    Returns:
        Tensor of shape (n,), 平滑后的 beta
    """
    return a * beta_prev_smooth + b * beta_curr_obs
# 设置环境变量
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'
torch.cuda.empty_cache()
np.random.seed(seed=0)
random.seed(0)
server_ip = '192.168.50.135'
server_port = 8083
s2= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 获取相机参数(GPU)
def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda:3')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K


# part1
def part_1(img,model_p1,frame_count,K):
    nms_kernel_size=3
    det_thresh=0.45
    t1=time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            out1=model_p1(img,
                is_training=False, 
                nms_kernel_size=int(nms_kernel_size),
                det_thresh=det_thresh,
                K=K)
            print(f"{frame_count}part1 处理耗时: {time.time()-t1:.4f}s")
            return out1,frame_count

# part2
def part_2(out_list,img,frame_count,model_p2,K):
    # token_out,mask_list,counts_list,K_det_list,loc_list,img_size,idx__list,scores_det_list,sign_parameter=False
    t1=time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            out2=model_p2(out_list[0],
                        out_list[1],
                        out_list[2],
                        out_list[3],
                        out_list[4],
                        out_list[5],
                        out_list[6],
                        out_list[7],
            sign_parameter=False)
            print(f"{frame_count}the rest推理耗时: {time.time()-t1:.4f}s")
            return out2,frame_count

# # 通过udp实时获取视频帧并进行预处理
# def img_reader_realtime_udp(img_queue,img_size,s,stop_event,device):
#     frame_count = 0
#     while not stop_event.is_set():
#         t1=time.time()
#         data, addr = s.recvfrom(400000)
#         print(f"{frame_count}接收图片耗时: {time.time()-t1:.4f}s")
#         nparr = np.frombuffer(data, np.uint8)
#         img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img_decode = cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB)
#         h, w, _ = img_decode.shape
#         aspect_ratio = w / h
#         if aspect_ratio > 1:  # Width > Height
#             new_w = img_size
#             new_h = int(img_size / aspect_ratio)
#         else:  # Height >= Width
#             new_h = img_size
#             new_w = int(img_size * aspect_ratio)

#         # Resize image while maintaining aspect ratio
#         img_cv = cv2.resize(img_decode, (new_w, new_h))

#         # Create a new image with padding
#         padded_img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)  # White padding
#         padded_img[:new_h, :new_w, :] = img_cv  # Place resized image in the top-left corner

#         # Normalize and go to torch
#         resize_img = normalize_rgb(padded_img)  

#         t2 = time.time()
#         print(f"{frame_count}预处理图片第一部分(接收，调整分辨率，规范化等)耗时: , {t2-t1:.4f}s")
#         x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
        
#         t3 = time.time()
#         print(f"{frame_count}预处理图片第二部分(cpu----->gpu)耗时: , {t3-t2:.4f}s")

#         img_queue.put((x,frame_count,padded_img))
#         print(f"\n{frame_count}接收图片全过程（包括放入图片队列）耗时: {time.time()-t1:.4f}s")
#         frame_count+=1

def img_reader_realtime_udp(img_queue, img_size, s, stop_event, device):
    frame_count = 0
    last_frame_number = -1
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            t1 = time.time()
            # 接收数据包
            data, addr = s.recvfrom(65535)  # 最大UDP包大小
            
            # 解析帧号（前4个字节）
            frame_number = struct.unpack('!I', data[:4])[0]
            image_data = data[4:]
            
            # 计算FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            
            # 打印帧号和FPS信息
            print(f'收到帧号: {frame_number}, FPS: {fps:.2f}')
            
            # 保存帧号信息不一致的日志
            if frame_number != last_frame_number + 1:
                print(f'帧号不连续: 上一帧 {last_frame_number}, 当前帧 {frame_number}')
            last_frame_number = frame_number
            
            # 将图像数据转换为OpenCV可以处理的格式
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_decode = cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB)
                
                # 调整图像大小和预处理（保持原有逻辑不变）
                h, w, _ = img_decode.shape
                aspect_ratio = w / h
                if aspect_ratio > 1:  # Width > Height
                    new_w = img_size
                    new_h = int(img_size / aspect_ratio)
                else:  # Height >= Width
                    new_h = img_size
                    new_w = int(img_size * aspect_ratio)

                # Resize image while maintaining aspect ratio
                img_cv = cv2.resize(img_decode, (new_w, new_h))

                # Create a new image with padding
                padded_img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)  # White padding
                padded_img[:new_h, :new_w, :] = img_cv  # Place resized image in the top-left corner

                # Normalize and go to torch
                resize_img = normalize_rgb(padded_img)  

                t2 = time.time()
                print(f"{frame_count}预处理图片第一部分(接收，调整分辨率，规范化等)耗时: {t2-t1:.4f}s")
                x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
                
                t3 = time.time()
                print(f"{frame_count}预处理图片第二部分(cpu----->gpu)耗时: {t3-t2:.4f}s")

                img_queue.put((x, frame_count, padded_img))
                print(f"\n{frame_count}接收图片全过程（包括放入图片队列）耗时: {time.time()-t1:.4f}s")
                
            except Exception as e:
                print(f'处理图像时出错: {e}')

    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        # 清理资源的代码应该在调用这个函数的上层进行
        pass


# part1处理
def part1_process(img_queue,out1_queue,stop_event,model_p1,K,out_part1_list,img_size):
    while not stop_event.is_set() or not img_queue.empty():
        if not img_queue.empty():
            t1=time.time()
            img,frame_count,img_pil_nopad = img_queue.get()  # 从队列中获取图像
            print(f"{frame_count}图片队列长度:", img_queue.qsize())
            t2=time.time()
            out1,frame_count=part_1(img,model_p1,frame_count,K)
            print(f"{frame_count}part1处理耗时: {time.time()-t1:.4f}s")
            if len(out_part1_list)!=3:
                out_part1_list.append(out1)
                continue
            else:
                # 滑动窗口
                t=time.time()
                out_part1_list.pop(0)
                out_part1_list.append(out1)
                try:
                    max_people=max(out_part1_list[0][0].shape[1],out_part1_list[1][0].shape[1],out_part1_list[2][0].shape[1])
                    # max_idx=max(out_part1_list[0][2].shape[0],out_part1_list[1][2].shape[0],out_part1_list[2][2].shape[0])
                except Exception as e:
                    print(f"\n\nerror: 帧{frame_count}没有检测到人")
                    continue
                # max_people=max(out_part1_list[0][0].shape[1],out_part1_list[1][0].shape[1],out_part1_list[2][0].shape[1])
                # max_idx=max(out_part1_list[0][2].shape[0],out_part1_list[1][2].shape[0],out_part1_list[2][2].shape[0])
                token_out1,inits1,inx_0_1,k_det_1,loc_1,counts_1,mask_1,scores_1,offset_1,scores_det_1=out_part1_list[0][0],out_part1_list[0][1],out_part1_list[0][2],out_part1_list[0][3],out_part1_list[0][4],out_part1_list[0][5],out_part1_list[0][6],out_part1_list[0][7],out_part1_list[0][8],out_part1_list[0][9]
                token_out2,inits2,inx_0_2,k_det_2,loc_2,counts_2,mask_2,scores_2,offset_2,scores_det_2=out_part1_list[1][0],out_part1_list[1][1],out_part1_list[1][2],out_part1_list[1][3],out_part1_list[1][4],out_part1_list[1][5],out_part1_list[1][6],out_part1_list[1][7],out_part1_list[1][8],out_part1_list[1][9]
                token_out3,inits3,inx_0_3,k_det_3,loc_3,counts_3,mask_3,scores_3,offset_3,scores_det_3=out_part1_list[2][0],out_part1_list[2][1],out_part1_list[2][2],out_part1_list[2][3],out_part1_list[2][4],out_part1_list[2][5],out_part1_list[2][6],out_part1_list[2][7],out_part1_list[2][8],out_part1_list[2][9]

                padding1 = (0, 0, 0, max_people-out_part1_list[0][0].shape[1])
                padding1_mask=(0,max_people-out_part1_list[0][0].shape[1])
                token_out1_padded = torch.nn.functional.pad(token_out1, padding1, value=0)
                # print(token_out1_padded.shape)
                mask_1_padded = torch.nn.functional.pad(mask_1, padding1_mask, value=0)


                padding2 = (0, 0, 0, max_people-out_part1_list[1][0].shape[1])
                padding2_mask=(0,max_people-out_part1_list[1][0].shape[1])  
                token_out2_padded = torch.nn.functional.pad(token_out2, padding2, value=0)
                # print(token_out2_padded.shape)
                mask_2_padded = torch.nn.functional.pad(mask_2, padding2_mask, value=0)

                padding3 = (0, 0, 0, max_people-out_part1_list[2][0].shape[1])
                padding3_mask=(0,max_people-out_part1_list[2][0].shape[1])
                token_out3_padded = torch.nn.functional.pad(token_out3, padding3, value=0)
                # print(token_out3_padded.shape)
                mask_3_padded = torch.nn.functional.pad(mask_3, padding3_mask, value=0)

                token_out = torch.stack([token_out1_padded, token_out2_padded, token_out3_padded], dim=1)
                mask_list = torch.stack([mask_1_padded, mask_2_padded, mask_3_padded], dim=1)
                counts_list=torch.stack([counts_1, counts_2, counts_3], dim=0)
                K_det_list=torch.cat([k_det_1, k_det_2, k_det_3], dim=0)
                loc_list=torch.cat([loc_1, loc_2, loc_3], dim=0)

                idx__list=[inx_0_1, inx_0_2, inx_0_3]
                scores_det_list=[scores_det_1, scores_det_2, scores_det_3]

                print(f"拼装耗时：{time.time()-t:.4f}s")
                out_list=[token_out,mask_list,counts_list,K_det_list,loc_list,img_size,idx__list,scores_det_list]
                out1_queue.put((out_list,img,frame_count,img_pil_nopad))
        else:
            time.sleep(0.001)  # 避免忙等

# part2处理
def part2_process(out1_queue,result_queue,stop_event,model_p2,K):
    while not stop_event.is_set() or not out1_queue.empty():
        if not out1_queue.empty():
            t1=time.time()
            out_list,img,frame_count,img_pil_nopad = out1_queue.get()  # 从队列中获取图像
            print(f"{frame_count}part1输出队列长度:", out1_queue.qsize())
            out2,frame_count=part_2(out_list,img,frame_count,model_p2,K)
            print(f"{frame_count}part2处理耗时: {time.time()-t1:.4f}s")
            result_queue.put((out2[2],img,frame_count,img_pil_nopad))
            # result_queue.put((out2[0],img,frame_count,img_pil_nopad))
        else:
            time.sleep(0.001)  # 避免忙等

# part2处理
def part2_process(out1_queue,result_queue,stop_event,model_p2,K):
    while not stop_event.is_set() or not out1_queue.empty():
        if not out1_queue.empty():
            t1=time.time()
            out_list,img,frame_count,img_pil_nopad = out1_queue.get()  # 从队列中获取图像
            print(f"{frame_count}part1输出队列长度:", out1_queue.qsize())
            out2,frame_count=part_2(out_list,img,frame_count,model_p2,K)
            print(f"{frame_count}part2处理耗时: {time.time()-t1:.4f}s")
            t3=time.time()
            # 高斯平滑
            poses_all=torch.stack([out2[0][0]['rotvec'],out2[1][0]['rotvec'],out2[2][0]['rotvec']],dim=0)
            trans_all=torch.stack([out2[0][0]['transl_pelvis'],out2[1][0]['transl_pelvis'],out2[2][0]['transl_pelvis']],dim=0)
            shape_all=torch.stack([out2[0][0]['shape'],out2[1][0]['shape'],out2[2][0]['shape']],dim=0)
            # print("================shape",poses_all.shape,trans_all.shape,shape_all.shape)
            poses_all_new,shape_all_new,trans_all_new=smplx_gs_smooth(poses_all,shape_all,trans_all)
            out2[0][0]['rotvec']=poses_all_new[0]
            out2[1][0]['rotvec']=poses_all_new[1]
            out2[2][0]['rotvec']=poses_all_new[2]
            out2[0][0]['transl_pelvis']=trans_all_new[0]
            out2[1][0]['transl_pelvis']=trans_all_new[1]
            out2[2][0]['transl_pelvis']=trans_all_new[2]
            out2[0][0]['shape']=shape_all_new[0]
            out2[1][0]['shape']=shape_all_new[1]
            out2[2][0]['shape']=shape_all_new[2]
            print(f"=======================================================高斯平滑耗时：{time.time()-t3:.4f}s")

            result_queue.put((out2[2],img,frame_count,img_pil_nopad))
            # result_queue.put((out2[0],img,frame_count,img_pil_nopad))
        else:
            time.sleep(0.001)  # 避免忙等


# 渲染函数
# def render_function(result_queue, stop_event, K_render, rvec, tvec):
#     prev_time = time.time()
#     fps = 0
#     while not stop_event.is_set() or not result_queue.empty():
#         if not result_queue.empty():
            
#             results, img,frame_count,img_pil_nopad = result_queue.get()
#             print(f"{frame_count}结果队列长度:", result_queue.qsize())
#             try:
#                 points_3d = results[0]['v3d'].cpu().numpy()
#                 # points_3d = results[2]['v3d'].cpu().numpy()
#                 # joints_2d = results[0]['j2d'].cpu().numpy()
#             except Exception as e:
#                 print(e)
#                 print(f"\n\nerror: 帧{frame_count}没有检测到人")
#                 points_3d = None
#                 # joints_2d=None
#                 continue

#             try:
#                 t1=time.time()
#                 data_for_send=str([results[0]['transl_pelvis'],results[0]['rotvec'],results[0]['shape']]).encode()
#                 s2.sendto(data_for_send, (server_ip, server_port))
#                 print("发送耗时：",time.time()-t1)

#             except Exception as e:
#                 print(e)
#             image = np.asarray(img_pil_nopad).copy()
#             image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K_render, None)
#             if points_3d is not None and points_3d.any():
#                print(f"\n\n帧{frame_count}检测到人！！！！！")
#                for point in points_2d:
#                    x, y = int(point[0][0]), int(point[0][1])
#                    if 0 <= x < 672 and 0 <= y < 672:
#                        cv2.drawMarker(image_, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
#             # if joints_2d is not None and joints_2d.any():
#             #     print(f"\n\n帧{frame_count}检测到人！！！！！")
#             #     for point in joints_2d:
#             #         x, y = int(point[0]), int(point[1])
#             #         if 0 <= x < 672 and 0 <= y < 672:
#             #             cv2.drawMarker(image_, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=5)
#             cv2.putText(image_, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             cv2.imshow("Projected 2D Points", image_)
#             key = cv2.waitKey(1)
#             if key == 27:
#                 stop_event.set()
#                 break
#             if cv2.getWindowProperty("Projected 2D Points", cv2.WND_PROP_VISIBLE) < 1:
#                 stop_event.set()
#                 break
#             current_time = time.time()
#             elapsed_time = current_time - prev_time
#             prev_time = current_time
#             fps = 1 / elapsed_time
            
#             print(f"\n{frame_count}渲染帧率: {fps:.4f}fps")
#         else:
#             time.sleep(0.001)

def render_function(result_queue, stop_event, K_render, rvec, tvec, udp_send_queue,transl_pelvis_prev,rotvec_prev,shape_prev,args_a,args_b,args_t_prev,args_t_new):
    prev_time = time.time()
    fps = 0

    while not stop_event.is_set() or not result_queue.empty():
        if not result_queue.empty():
            results, img, frame_count, img_pil_nopad = result_queue.get()
            print(f"{frame_count}结果队列长度:", result_queue.qsize())
            try:
                points_3d = results[0]['v3d'].cpu().numpy()
            except Exception as e:
                print(e)
                print(f"\n\nerror: 帧{frame_count}没有检测到人")
                continue
            print("====================================================",results[0]['transl_pelvis'].shape, results[0]['rotvec'].shape, results[0]['shape'].shape)
            # torch.Size([1, 3]) torch.Size([53, 3]) torch.Size([10])
            # 放入 UDP 发送队列（非阻塞）
            try:
                transl_pelvis_curr=results[0]['transl_pelvis']
                rotvec_curr=results[0]['rotvec']
                shape_curr=results[0]['shape']

                # pose
                q0 = axisangle_to_quat_torch(rotvec_prev)
                q1 = axisangle_to_quat_torch(rotvec_curr)
                a, b = args_a,args_b
                t_prev, t_curr = args_t_prev, args_t_new
                t_new = (a * t_curr + b * t_prev) / (a + b)
                # q_smooth = slerp_batch(q0, q1, torch.tensor(t_new))
                q_smooth = slerp_batch(q0, q1, t_new)
                rotvec_curr_new = quat_to_axisangle_torch(q_smooth)
                rotvec_prev=rotvec_curr

                # transl
                transl_pelvis_curr_new = smooth_translation(transl_pelvis_prev, transl_pelvis_curr, a=args_a, b=args_b)
                transl_pelvis_prev=transl_pelvis_curr

                # shape
                shape_curr_new = smooth_shape_linear(shape_prev, shape_curr, a=args_a, b=args_b)
                shape_prev=shape_curr

                print(transl_pelvis_curr_new.shape, rotvec_curr_new.shape, shape_curr_new.shape)


                data_for_send = str([
                    transl_pelvis_curr_new,
                    rotvec_curr_new,
                    shape_curr_new
                ]).encode()
                udp_send_queue.put(data_for_send)
            except Exception as e:
                print("准备发送数据异常:", e)

            image = np.asarray(img_pil_nopad).copy()
            image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #==============渲染部分开始==========================
            points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K_render, None)

            if points_3d is not None and points_3d.any():
                print(f"\n\n帧{frame_count}检测到人！！！！！")
                for point in points_2d:
                    x, y = int(point[0][0]), int(point[0][1])
                    if 0 <= x < 672 and 0 <= y < 672:
                        cv2.drawMarker(image_, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
            #===============渲染部分结束=========================

            cv2.putText(image_, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow("Projected 2D Points", image_)
            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty("Projected 2D Points", cv2.WND_PROP_VISIBLE) < 1:
                stop_event.set()
                break

            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = current_time
            fps=0.9 * fps + 0.1 * (1 / elapsed_time)
            # fps = 1 / elapsed_time
            print(f"\n{frame_count}渲染帧率: {fps:.4f}fps")
        else:
            time.sleep(0.001)


# def render_function(result_queue, stop_event, K_render, rvec, tvec, udp_send_queue):
#     prev_time = time.time()
#     fps = 0
#     # 创建OpenCV窗口一次，避免重复创建开销
#     cv2.namedWindow("Projected 2D Points", cv2.WINDOW_NORMAL)
    
#     while not stop_event.is_set() or not result_queue.empty():
#         if not result_queue.empty():
#             results, img, frame_count, img_pil_nopad = result_queue.get()
#             print(f"{frame_count}结果队列长度:", result_queue.qsize())
            
#             try:
#                 points_3d = results[0]['v3d'].cpu().numpy()
#             except Exception as e:
#                 print(e)
#                 print(f"\n\nerror: 帧{frame_count}没有检测到人")
#                 continue

#             # 放入 UDP 发送队列（非阻塞）
#             try:
#                 data_for_send = str([
#                     results[0]['transl_pelvis'],
#                     results[0]['rotvec'],
#                     results[0]['shape']
#                 ]).encode()
#                 udp_send_queue.put(data_for_send)
#             except Exception as e:
#                 print("准备发送数据异常:", e)

#             # 优化图像转换和复制操作
#             image = np.asarray(img_pil_nopad)
#             image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             # if image.ndim == 3 and image.shape[2] == 3:
#             #     # 直接处理BGR图像，避免颜色空间转换
#             #     image_ = image.copy()
#             # else:
#             #     # 只有必要时才进行颜色转换
#             #     image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             #==============优化渲染部分开始==========================
#             if points_3d is not None and points_3d.size > 0:
#                 # 批量投影3D点到2D平面
#                 points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K_render, None)
#                 points_2d = np.int32(points_2d.reshape(-1, 2))
                
#                 # 过滤超出图像边界的点
#                 valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < 672) & \
#                              (points_2d[:, 1] >= 0) & (points_2d[:, 1] < 672)
#                 valid_points = points_2d[valid_mask]
                
#                 if len(valid_points) > 0:
#                     print(f"\n\n帧{frame_count}检测到人！！！！！")
#                     # 使用numpy数组批量绘制标记
#                     for x, y in valid_points:
#                         cv2.drawMarker(image_, (x, y), (255, 255, 255), 
#                                       markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
#             #===============优化渲染部分结束=========================

#             # 使用更高效的FPS计算方法
#             current_time = time.time()
#             elapsed_time = current_time - prev_time
#             prev_time = current_time
#             fps = 0.9 * fps + 0.1 * (1 / elapsed_time)  # 低通滤波平滑FPS计算
            
#             # 添加显示信息
#             cv2.putText(image_, f"FPS: {fps:.2f}", (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
#             # 优化窗口显示和按键处理
#             cv2.imshow("Projected 2D Points", image_)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or cv2.getWindowProperty("Projected 2D Points", cv2.WND_PROP_VISIBLE) < 1:
#                 stop_event.set()
#                 break

#             print(f"\n{frame_count}渲染帧率: {fps:.4f}fps")
#         else:
#             # 优化空闲时的CPU使用率
#             time.sleep(0.001)
    


def udp_sender_thread_func(send_queue, stop_event, server_ip, server_port):
    while not stop_event.is_set() or not send_queue.empty():
        if not send_queue.empty():
            try:
                data = send_queue.get()
                t1 = time.time()
                s2.sendto(data, (server_ip, server_port))
                print("发送耗时：", time.time() - t1)
            except Exception as e:
                print("UDP 发送异常:", e)
        else:
            time.sleep(0.001)  # 避免忙等


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path_p1", type=str, default='/home/zzk/projects/multi-hmr/ckpt/model/00049.pt')
    parser.add_argument("--ckpt_path_p2", type=str, default='/home/zzk/projects/multi-hmr/ckpt/smooth/00049_smooth.pt')
    parser.add_argument("--model_name", type=str, default='multi_hmr_temporal_b_672')
    parser.add_argument("--img_size", type=int, default=672)
    parser.add_argument("--ip", type=str, default='')  # 默认监听所有ip地址
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--backbone", type=str, default='dinov2_vitb14')
    parser.add_argument("--queue_size", type=int, default=5)
    
    args=parser.parse_args()
    # 创建UDP套接字，绑定到指定的IP地址和端口号
    address=(args.ip, args.port)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(address)
    

    # 创建队列
    img_queue = queue.Queue(maxsize=args.queue_size)        # 限制图像队列大小
    out1_queue = queue.Queue(maxsize=args.queue_size)       # 限制part1队列大小
    out2_queue = queue.Queue(maxsize=args.queue_size)       # 限制part2队列大小
    result_queue = queue.Queue(maxsize=args.queue_size)     # 用于存储最终处理结果
    udp_send_queue = queue.Queue(maxsize=100)
    # GPU初始化
    device = torch.device(args.device)


    # 加载模型
    ckpt1 = torch.load(args.ckpt_path_p1,map_location=torch.device('cuda:0'))
    ckpt2 = torch.load(args.ckpt_path_p2,map_location=torch.device('cuda:0'))
    model_p1=Model_temporal(img_size=args.img_size,backbone=args.backbone).to(args.device)
    model_p2=TemporalSmoothing().to(args.device)
    model_p1.load_state_dict(ckpt1['model_state_dict'], strict=False)
    model_p2.load_state_dict(ckpt2['model_state_dict'], strict=False)
    model_p1.eval()
    model_p2.eval()

    # 相机内参
    p_x, p_y = None, None
    K = get_camera_parameters(args.img_size, fov=60, p_x=p_x, p_y=p_y,device=args.device)  # 获取相机内参
    K_render = K.reshape(3,3).cpu().numpy()                             # 获取相机内参（渲染用）

    # 定义旋转矩阵和位移向量
    rvec = np.zeros((3, 1), dtype=np.float32)  # 旋转向量
    tvec = np.zeros((3, 1), dtype=np.float32)  # 位移向量


    # 前一帧结果
    transl_pelvis_prev,rotvec_prev,shape_prev=torch.zeros(1,3).to(args.device),torch.zeros(53,3).to(args.device),torch.zeros(10,).to(args.device)

    # 一些参数
    args_a=torch.tensor(0.6).to(args.device)
    args_b=torch.tensor(0.4).to(args.device)
    args_t_prev=torch.tensor(0.0).to(args.device)
    args_t_new=torch.tensor(1.0).to(args.device)

    # 用于停止标志的事件
    stop_event = threading.Event()


    # 创建各类线程
    reader_thread = threading.Thread(target=img_reader_realtime_udp, args=(img_queue,args.img_size,s,stop_event,args.device))
    part1_thread = threading.Thread(target=part1_process, args=(img_queue,out1_queue,stop_event,model_p1,K, [],args.img_size))
    part2_thread = threading.Thread(target=part2_process, args=(out1_queue,result_queue,stop_event,model_p2,K))
    render_thread = threading.Thread(target=render_function, args=(result_queue, stop_event, K_render,rvec, tvec, udp_send_queue,transl_pelvis_prev,rotvec_prev,shape_prev,args_a,args_b,args_t_prev,args_t_new))
    udp_thread = threading.Thread(target=udp_sender_thread_func, args=(udp_send_queue, stop_event, server_ip, server_port))

    # 启动所有线程
    reader_thread.start()
    part1_thread.start()
    part2_thread.start()
    render_thread.start()
    udp_thread.start()

    # 等待所有线程结束
    reader_thread.join()
    part1_thread.join()
    part2_thread.join()
    render_thread.join()
    udp_thread.join()

    cv2.destroyAllWindows()
#     s.close()






                
            
