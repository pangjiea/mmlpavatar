from argparse import ArgumentParser
import argparse
import os
import pickle
import queue
import socket
import struct
import threading
from tqdm.auto import tqdm
import torch
import numpy as np
import yaml
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import time
import cv2
from models.sat_model import *

def update_args(args, cfg_path):
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
        args_dict = vars(args)
        args_dict.update(config)
        args = argparse.Namespace(**args_dict)
    return args

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

def smooth_verts_linear(verts_prev_smooth: torch.Tensor,
                        verts_curr_obs: torch.Tensor,
                        a: float = 0.8,
                        b: float = 0.2) -> torch.Tensor:
    return a * verts_prev_smooth + b * verts_curr_obs


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


def process_img0(img,device,mean,std,meta_data=None,input_size=1288, patch_size=56, use_color_jitter=False):
    t1 = time.time()

    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_size = torch.tensor(img.shape[:2], device=device)
    target=[{'img_size': img_size}]
    pad_h = math.ceil(new_h / patch_size) * patch_size
    pad_w = math.ceil(new_w / patch_size) * patch_size

    # Zero padding（避免 Image/PIL 转换）
    padded_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w, :] = img

    # Convert BGR to RGB and normalize in-place
    img_tensor = torch.from_numpy(padded_img[:, :, ::-1].copy()).float() / 255.0  # HWC, RGB
    img_tensor = img_tensor.permute(2, 0, 1).to(device)  # CHW, float32, to device

    if use_color_jitter:
        # Simple approximation of ColorJitter on GPU (faster than PIL transforms)
        brightness = (1.0 + (torch.rand(1, device=device) - 0.5) * 0.4)
        contrast = (1.0 + (torch.rand(1, device=device) - 0.5) * 0.4)
        img_tensor = img_tensor * brightness
        img_tensor = (img_tensor - img_tensor.mean()) * contrast + img_tensor.mean()
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    # Normalize
    img_tensor = (img_tensor - mean) / std

    print(f"Resize+Preprocess time: {time.time() - t1:.4f} seconds")
    return img_tensor, target

def process_img(img, device, mean, std, meta_data=None, input_size=1288, patch_size=56, use_color_jitter=False):
    t1 = time.time()

    # 保存原始图像尺寸
    original_h, original_w = img.shape[:2]

    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_size = torch.tensor(img.shape[:2], device=device)
    target = [{'img_size': img_size}]
    pad_h = math.ceil(new_h / patch_size) * patch_size
    pad_w = math.ceil(new_w / patch_size) * patch_size

    # Zero padding（避免 Image/PIL 转换）
    padded_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w, :] = img

    # Convert BGR to RGB and normalize in-place
    img_tensor = torch.from_numpy(padded_img[:, :, ::-1].copy()).float() / 255.0  # HWC, RGB
    img_tensor = img_tensor.permute(2, 0, 1).to(device)  # CHW, float32, to device

    if use_color_jitter:
        # Simple approximation of ColorJitter on GPU (faster than PIL transforms)
        brightness = (1.0 + (torch.rand(1, device=device) - 0.5) * 0.4)
        contrast = (1.0 + (torch.rand(1, device=device) - 0.5) * 0.4)
        img_tensor = img_tensor * brightness
        img_tensor = (img_tensor - img_tensor.mean()) * contrast + img_tensor.mean()
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    # Normalize
    img_tensor = (img_tensor - mean) / std

    # 反归一化
    unnormalized_img = img_tensor * std + mean
    unnormalized_img = unnormalized_img.permute(1, 2, 0).cpu().numpy() * 255.0  # CHW to HWC, RGB to BGR
    unnormalized_img = unnormalized_img.astype(np.uint8)[:, :, ::-1]  # RGB to BGR

    # 恢复边界
    ori_img = unnormalized_img.copy()

    print(f"Resize+Preprocess time: {time.time() - t1:.4f} seconds")
    return img_tensor, target, ori_img


def img_reader_realtime_udp(img_queue, s, stop_event, device,mean,std, input_size=1288):
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
                processed_img,target,ori_img = process_img(img_decode, device=device,mean=mean,std=std, input_size=input_size)              
                img_queue.put((processed_img, frame_count, target,ori_img))
                print(f"\n{frame_count}接收图片和预处理全过程（包括放入图片队列）耗时: {time.time()-t1:.4f}s")
                
            except Exception as e:
                print(f'处理图像时出错: {e}')

    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        pass

def model_process(img_queue,output_queue,stop_event,model):
    while not stop_event.is_set() or not img_queue.empty():
        if not img_queue.empty():
            t1=time.time()
            processed_img, frame_count, target, ori_img = img_queue.get()
            with torch.no_grad():
                outputs = model([processed_img], target)

            output_queue.put((outputs, frame_count, target, ori_img))
            print(f"\n{frame_count}推理全程耗时: {time.time()-t1:.4f}s")

        else:
            time.sleep(0.001)  # 避免忙等
def render_function(output_queue, stop_event,rvec, tvec, udp_send_queue,transl_pelvis_prev,rotvec_prev,shape_prev,verts_prev,args_a,args_b,args_t_prev,args_t_new):
    prev_time = time.time()
    fps = 0

    while not stop_event.is_set() or not output_queue.empty():
        if not output_queue.empty():
            outputs, frame_count, target, ori_img= output_queue.get()
            h,w= ori_img.shape[:2]
            try:
                select_queries_idx = torch.where(outputs['pred_confs'][0] > 0.3)[0]
                pred_verts = outputs['pred_verts'][0][select_queries_idx].detach().cpu().numpy()
                if(len(pred_verts)>0):
                    pred_verts = pred_verts.reshape(6890,3)
                else:

                    print(f"帧 {frame_count} 中未检测到有效的 pred_verts，跳过处理")
                    continue
                verts_curr=outputs['pred_verts'][0][select_queries_idx]
                # print(pred_verts.shape)
            except Exception as e:
                print(e)
                print(f"\n\nerror: 帧{frame_count}检测出错！！！",e)
                continue
            print("====================================================",outputs['pred_transl'][0][select_queries_idx].shape, outputs['pred_poses'][0][select_queries_idx].shape, outputs['pred_betas'][0][select_queries_idx].shape)
            # torch.Size([1, 3]) torch.Size([24, 3]) torch.Size([10])
            # 放入 UDP 发送队列（非阻塞）
            try:
                if (len(outputs['pred_transl'][0][select_queries_idx])>0):
                    transl_pelvis_curr=outputs['pred_transl'][0][select_queries_idx].reshape(1,3)
                    rotvec_curr=outputs['pred_poses'][0][select_queries_idx].reshape(24,3)
                    shape_curr= outputs['pred_betas'][0][select_queries_idx].reshape(10,)
                else:

                    continue


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


                # verts
                print(verts_prev,verts_curr)
                verts_curr_new =smooth_verts_linear(verts_prev,verts_curr, a=args_a, b=args_b)
                pred_verts=verts_curr_new
                verts_prev=verts_curr

                # print(transl_pelvis_curr_new.shape, rotvec_curr_new.shape, shape_curr_new.shape)

                tensor_22x3 = rotvec_curr_new[:22, :]
                tensor_53x3 = torch.zeros(53, 3)
                tensor_53x3[:22, :] = tensor_22x3
                rotvec_curr_new=tensor_53x3

                data_for_send = str([
                    transl_pelvis_curr_new,
                    rotvec_curr_new,
                    shape_curr_new
                ]).encode()
                udp_send_queue.put(data_for_send)
            except Exception as e:
                print("准备发送数据异常:", e)
                # continue
            # print(verts_curr_new)
            pred_verts=verts_curr_new.cpu().numpy()
            image_ = ori_img
            K_render=outputs['pred_intrinsics'][0].reshape(3,3).detach().cpu().numpy()
            #==============渲染部分开始==========================
            points_2d, _ = cv2.projectPoints(pred_verts, rvec, tvec, K_render, None)

            if pred_verts is not None and pred_verts.any():
                print(f"\n\n帧{frame_count}检测到人！！！！！")
                for point in points_2d:
                    x, y = int(point[0][0]), int(point[0][1])
                    if 0 <= x < w and 0 <= y < h:
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
if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--ckpt_path", type=str, default='/home/zzk/projects/sat-hmr/weights/sat_hmr/sat_644.pth')
    parser.add_argument("--img_size", type=int, default=1288)
    parser.add_argument("--ip", type=str, default='') 
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--backbone", type=str, default='dinov2_vitb14')
    parser.add_argument("--queue_size", type=int, default=5)
    # 将 server_ip 默认改为 'auto'，运行时解析为本机 IP
    parser.add_argument("--server_ip", type=str, default='100.97.28.58') 
    parser.add_argument("--server_port", type=int, default=8082)
    parser.add_argument("--mode", type=str, default='infer')
    args = parser.parse_args()
    args = update_args(args, os.path.join('configs', 'run', 'demo.yaml'))
    args = update_args(args, os.path.join('configs', 'models', f'{args.model}.yaml'))

    # 解析本机 IP：当 --server_ip 为 'auto' 或空时自动取主网卡 IP
    def _resolve_local_ip():
        try:
            tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 不会真实发包，目的是让内核选择出站网卡，从而拿到本机出站 IP
            tmp.connect(('8.8.8.8', 80))
            ip = tmp.getsockname()[0]
            tmp.close()
            return ip
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return '127.0.0.1'

    if (not args.server_ip) or (str(args.server_ip).lower() in ('auto', '0.0.0.0')):
        resolved_ip = _resolve_local_ip()
        print(f"[info] Resolved server_ip to local IP: {resolved_ip}")
        args.server_ip = resolved_ip


    address=(args.ip, args.port)
    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s2= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s1.bind(address)

    # 创建队列
    img_queue = queue.Queue(maxsize=args.queue_size)
    output_queue = queue.Queue(maxsize=args.queue_size)
    send_queue = queue.Queue(maxsize=args.queue_size)

    # 用于停止标志的事件
    stop_event = threading.Event()

    # GPU初始化
    device = torch.device(args.device)

    # 加载模型
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model(
        encoder,
        decoder,
        num_queries=args.num_queries,
        input_size=args.input_size,
        sat_cfg=args.sat_cfg,
        dn_cfg=args.dn_cfg,
        train_pos_embed=getattr(args,'train_pos_embed',True)
    )

    # 前一帧结果，注意是smpl而不是smplx
    transl_pelvis_prev,rotvec_prev,shape_prev=torch.zeros(1,3).to(device),torch.zeros(24,3).to(device),torch.zeros(10,).to(device)
    verts_prev=torch.ones(6890,3).to(device)

    # 一些参数
    args_a=torch.tensor(0.6).to(device)
    args_b=torch.tensor(0.4).to(device)
    args_t_prev=torch.tensor(0.0).to(device)
    args_t_new=torch.tensor(1.0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    rvec = np.zeros((3, 1), dtype=np.float32)  # 旋转向量
    tvec = np.zeros((3, 1), dtype=np.float32)  # 位移向量

    # 加载权重同时将模型放到gpu上
    state_dict = torch.load(args.pretrain_path, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    img_reader_thread= threading.Thread(target=img_reader_realtime_udp, args=(img_queue, s1, stop_event, device, mean, std, args.img_size))
    process_thread = threading.Thread(target=model_process, args=(img_queue, output_queue, stop_event, model))
    render_thread = threading.Thread(target=render_function, args=(output_queue, stop_event,rvec, tvec, send_queue, transl_pelvis_prev, rotvec_prev, shape_prev, verts_prev,args_a, args_b, args_t_prev, args_t_new))
    udp_thread = threading.Thread(target=udp_sender_thread_func, args=(send_queue, stop_event, args.server_ip, args.server_port))


    # 启动线程
    img_reader_thread.start()
    process_thread.start()
    render_thread.start()
    udp_thread.start()

    # 等待所有线程结束
    img_reader_thread.join()
    process_thread.join()
    render_thread.join()
    udp_thread.join()

    cv2.destroyAllWindows()


    # original_img = cv2.imread('/home/zzk/projects/sat-hmr/test.jpg')
    # # processed_img,target = process_img0(original_img, device=device,mean=mean,std=std, input_size=1288)
    # # print(type(target))
    # # with torch.no_grad():
    # #     outputs = model([processed_img], target)

    # # t= time.time()
    # # img_size = target[0]['img_size'].detach().cpu().int().numpy()
    # # ori_img = tensor_to_BGR(unNormalize(processed_img).cpu())
    # # ori_img[img_size[0]:,:,:] = 255
    # # ori_img[:,img_size[1]:,:] = 255
    # # ori_img[img_size[0]:,img_size[1]:,:] = 255
    # # ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
    


    # processed_img,target,ori_img = process_img(original_img, device=device,mean=mean,std=std, input_size=1288)
    # with torch.no_grad():
    #     outputs = model([processed_img], target)

    # t = time.time()
    # img_size = target[0]['img_size'].detach().cpu().int().numpy()
    # select_queries_idx = torch.where(outputs['pred_confs'][0] > 0.3)[0]
    # pred_verts = outputs['pred_verts'][0][select_queries_idx].detach().cpu().numpy()
    # print(f"处理时间: {time.time() - t:.4f}秒")
    # smpl_layer = model.human_model
    # colors = get_colors_rgb(len(pred_verts))
    
    # pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
    #                         verts = pred_verts,
    #                         smpl_faces = smpl_layer.faces,
    #                         cam_intrinsics = outputs['pred_intrinsics'][0].reshape(3,3).detach().cpu(),
    #                         colors=colors)[:img_size[0],:img_size[1]]
    

    

    # # 写入文件
    # cv2.imwrite("/home/zzk/projects/sat-hmr/test_result.png",pred_mesh_img)
    



    
