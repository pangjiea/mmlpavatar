#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from os import path
import torch
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import numpy as np
import random
import pickle
import copy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from scene.gaussian_model import GaussianModel
from scene.scene import Scene
from scene.dataset import data_to_cam
from scene.net_vis import Visualizer
from utils.config_utils import Config
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, psnr, lpips_loss, dxyz_smooth_loss, gaussian_scaling_loss
from utils.image_utils import crop_image, calc_face_bbox, calc_face_bbox_smplx, calc_face_mask_smplx
from omegaconf import OmegaConf
import imageio.v3 as iio
import numpy as np
from utils.smpl_utils import smpl

def training(args: Config):
    # Safe config access with defaults for newly added keys
    def cfg(key, default):
        v = OmegaConf.select(args, key)
        return default if v is None else v

    gaussians = GaussianModel()
    scene = Scene(args, gaussians)    
    gaussians.training_setup(args, scene.scene_scale)

    visualizer = Visualizer(in_training=True)
    visualizer.net_init(args.ip, args.port)
    visualizer.gaussians = gaussians
    visualizer.load_cams_poses(args.out_dir)

    background = torch.as_tensor(args.background).float().cuda()

    ema_vis_loss, ema_lpips_loss = 0.0, 0.0
    first_iter = 0
    progress_bar = tqdm(range(0, args.iterations), initial=first_iter, desc="TP")
    first_iter += 1
    trainloader_iter = iter(scene.trainloader)
    
    for iteration in range(first_iter, args.iterations + 1):     
        if iteration % 30 == 0:
            visualizer.is_send_initial_data = True
            visualizer.visualizing()
        
        try:
            cam = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(scene.trainloader)
            cam = next(trainloader_iter)

        cam = data_to_cam(cam)
        # Ensure async H2D transfers from dataset stream are complete before using tensors
        torch.cuda.synchronize()
        bg = torch.rand(3, device="cuda") if args.random_background else background

        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']
        # 从cam获取expression和从pose中提取jaw_pose、leye_pose、reye_pose
        # pose结构: [global_orient(3) + body_pose(63) + jaw_pose(3) + leye_pose(3) + reye_pose(3) + left_hand_pose(45) + right_hand_pose(45)]
        gaussians.expression = cam['expression']           # expression: 10维，单独在cam中
        gaussians.jaw_pose = gaussians.smpl_poses[66:69]    # jaw_pose: 3维
        gaussians.leye_pose = gaussians.smpl_poses[69:72]   # leye_pose: 3维
        gaussians.reye_pose = gaussians.smpl_poses[72:75]   # reye_pose: 3维

        image, alpha, info = gaussians.render(cam, background=bg)
        image = torch.clamp(image, 0, 1)
        image_gt, mask, mask_boundary = cam['image'], cam['mask'], cam['mask_boundary']
        image_gt[~mask] = bg
        image_gt[mask_boundary] = bg
        image[mask_boundary] = bg

        l1loss = l1_loss(image, image_gt)
        dxyzsmoothloss = dxyz_smooth_loss(gaussians) * args.lambda_dxyz_smooth

        random_patch_flag = False if iteration < args.iteration_lpips_random_patch else True
        image_crop, image_gt_crop = crop_image(bg, mask, 512, random_patch_flag, image.permute(2,0,1), image_gt.permute(2,0,1))
        if iteration > args.iteration_lpips: lpipsloss = lpips_loss(image_crop.permute(1,2,0), image_gt_crop.permute(1,2,0)) * args.lambda_lpips
        else: lpipsloss = torch.tensor(0) 

        # Face-region losses (coarse ROI based on silhouette bbox)
        face_l1loss = torch.tensor(0.0, device=image.device)
        face_lpipsloss = torch.tensor(0.0, device=image.device)
        face_bbox_for_debug = None
        face_mask_debug = None
        enable_face_loss = cfg('enable_face_loss', False)
        lambda_face_l1 = cfg('lambda_face_l1', 0.0)
        lambda_face_lpips = cfg('lambda_face_lpips', 0.0)
        face_mask_method = cfg('face_mask_method', 'smplx')
        face_roi_top_frac = cfg('face_roi_top_frac', 0.15)
        face_roi_height_frac = cfg('face_roi_height_frac', 0.22)
        face_roi_width_frac = cfg('face_roi_width_frac', 0.35)
        face_min_size = cfg('face_min_size', 32)
        face_smplx_radius_scale = cfg('face_smplx_radius_scale', 2.2)
        face_smplx_ax_min_frac = cfg('face_smplx_ax_min_frac', 0.25)
        face_smplx_ax_max_frac = cfg('face_smplx_ax_max_frac', 2.0)

        if enable_face_loss and (lambda_face_l1 > 0 or lambda_face_lpips > 0):
            try:
                # Build face bbox via chosen method
                face_bbox = None
                if face_mask_method == 'smplx':
                    face_mask_np, face_bbox = calc_face_mask_smplx(
                        pose_vec=gaussians.smpl_poses,
                        beta_vec=cam['beta'],
                        expression_vec=gaussians.expression,
                        jaw_pose_vec=gaussians.jaw_pose,
                        Rh=gaussians.Rh,
                        Th=gaussians.Th,
                        K=cam['K'],
                        w2c=cam['w2c'],
                        image_hw=(cam['height'], cam['width']),
                        radius_scale=face_smplx_radius_scale,
                        ax_min_frac=face_smplx_ax_min_frac,
                        ax_max_frac=face_smplx_ax_max_frac,
                        min_size=face_min_size,
                    )
                    face_mask_debug = face_mask_np
                elif face_mask_method == 'heuristic':
                    face_bbox = calc_face_bbox(mask, face_roi_top_frac, face_roi_height_frac,
                                               face_roi_width_frac, face_min_size)
                face_bbox_for_debug = face_bbox
                # TODO: detector method can be added later
                if face_mask_method == 'smplx' and face_mask_np is not None:
                    # Intersect with dataset silhouette to avoid spill to background
                    m_full = cam['mask'].detach().cpu().numpy().astype(np.uint8) * 255
                    face_mask_np = ((face_mask_np > 0) & (m_full > 0)).astype(np.uint8) * 255
                    face_mask_debug = face_mask_np
                    # Masked L1 over head region
                    mask_t = torch.from_numpy(face_mask_np > 0).to(image.device)
                    valid = mask_t.any()
                    if valid and lambda_face_l1 > 0:
                        diff = torch.abs(image - image_gt)
                        mask3 = mask_t.unsqueeze(-1).expand_as(diff)
                        face_l1loss = (diff[mask3].mean()) * lambda_face_l1
                    # LPIPS strictly inside face mask: set outside to same background, then crop bbox
                    if valid and lambda_face_lpips > 0 and iteration > args.iteration_lpips:
                        ys, xs = torch.nonzero(mask_t, as_tuple=True)
                        t, b = int(ys.min().item()), int(ys.max().item()) + 1
                        l, r = int(xs.min().item()), int(xs.max().item()) + 1
                        face_pred_hwc = image[t:b, l:r].clone()
                        face_gt_hwc = image_gt[t:b, l:r].clone()
                        patch_mask = mask_t[t:b, l:r].unsqueeze(-1)
                        # background color already defined as `bg` (3,) CUDA tensor
                        face_pred_hwc = torch.where(patch_mask, face_pred_hwc, bg[None, None, :])
                        face_gt_hwc = torch.where(patch_mask, face_gt_hwc, bg[None, None, :])
                        if face_pred_hwc.shape[0] > 0 and face_pred_hwc.shape[1] > 0:
                            face_lpipsloss = lpips_loss(face_pred_hwc, face_gt_hwc) * lambda_face_lpips
                elif face_bbox is not None:
                    # Fallback to bbox-based region
                    l, t, r, b = face_bbox
                    if lambda_face_l1 > 0:
                        face_pred = image[t:b, l:r]
                        face_gt = image_gt[t:b, l:r]
                        if face_pred.numel() > 0:
                            face_l1loss = l1_loss(face_pred, face_gt) * lambda_face_l1
                    if lambda_face_lpips > 0 and iteration > args.iteration_lpips:
                        face_pred_hwc = image[t:b, l:r]
                        face_gt_hwc = image_gt[t:b, l:r]
                        if face_pred_hwc.shape[0] > 0 and face_pred_hwc.shape[1] > 0:
                            face_lpipsloss = lpips_loss(face_pred_hwc, face_gt_hwc) * lambda_face_lpips
            except Exception as e:
                # Keep training robust if face ROI fails
                print(f"[warn] face loss skipped: {e}")

        scaling_loss = args.lambda_scaling * gaussian_scaling_loss(gaussians.get_cano_scaling, args.scaling_threshold)

        loss = l1loss + lpipsloss + dxyzsmoothloss + scaling_loss + face_l1loss + face_lpipsloss

        loss.backward()

        # log part
        ema_vis_loss = 0.4 * l1loss.item() + 0.6 * ema_vis_loss
        ema_lpips_loss = 0.4 * lpipsloss.item() + 0.6 * ema_lpips_loss
        if iteration % 10 == 0:
            progress_bar.set_postfix({'l1': f'{ema_vis_loss:.{4}f}' ,'lpips': f'{ema_lpips_loss:.{4}f}'})
            progress_bar.update(10)
        if iteration == args.iterations:
            progress_bar.close()
        if iteration == args.iteration_sh_degree:
            gaussians.sh_degree += 1
            print(f'SH degree: {gaussians.sh_degree}')

        loss_dict = dict(l1_loss=l1loss, lpips_loss=lpipsloss, dxyzsmooth_loss=dxyzsmoothloss, scaling_loss=scaling_loss,
                         face_l1=face_l1loss, face_lpips=face_lpipsloss)
        training_report(scene, gaussians, iteration, args.test_iterations, loss_dict, background)

        # optimizer step
        gaussians.optimizer_step()

        if iteration == args.iteration_dxyz_basis:
            gaussians.is_dxyz_bs = True
            print(f'[ITER {iteration}] Control point basis')

        if iteration == args.iteration_gsparam_basis:
            gaussians.is_gsparam_bs = True
            print(f'[ITER {iteration}] Gaussian property basis')

        # Debug: save face mask + GT + Rendered every 100 iterations
        if iteration % 100 == 0:
            try:
                H, W = cam['height'], cam['width']
                # Build a visualization mask
                vis_mask = np.zeros((H, W), dtype=np.uint8)
                if face_mask_method == 'smplx':
                    if face_mask_debug is None:
                        face_mask_debug, _ = calc_face_mask_smplx(
                            pose_vec=gaussians.smpl_poses,
                            beta_vec=cam['beta'],
                            expression_vec=gaussians.expression,
                            jaw_pose_vec=gaussians.jaw_pose,
                            Rh=gaussians.Rh,
                            Th=gaussians.Th,
                            K=cam['K'],
                            w2c=cam['w2c'],
                            image_hw=(H, W),
                            radius_scale=face_smplx_radius_scale,
                            ax_min_frac=face_smplx_ax_min_frac,
                            ax_max_frac=face_smplx_ax_max_frac,
                            min_size=face_min_size,
                        )
                    if face_mask_debug is not None:
                        # Intersect with silhouette for visualization
                        m_full = cam['mask'].detach().cpu().numpy().astype(np.uint8) * 255
                        vis_mask = ((face_mask_debug > 0) & (m_full > 0)).astype(np.uint8) * 255
                else:
                    if face_bbox_for_debug is None:
                        face_bbox_for_debug = calc_face_bbox(mask, face_roi_top_frac, face_roi_height_frac,
                                                             face_roi_width_frac, face_min_size)
                    if face_bbox_for_debug is not None:
                        l, t, r, b = face_bbox_for_debug
                        vis_mask[t:b, l:r] = 255

                # Convert tensors to uint8 RGB
                pred_np = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                gt_np = (image_gt.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                mask_rgb = np.stack([vis_mask, vis_mask, vis_mask], axis=-1)

                # Compose horizontally
                canvas = np.concatenate([pred_np, gt_np, mask_rgb], axis=1)
                debug_dir = path.join(args.out_dir, 'debug_face')
                os.makedirs(debug_dir, exist_ok=True)
                save_name = f"it_{iteration:06d}_cam{cam['cam_id']:02d}_frame{cam['frame_id']:06d}.png"
                iio.imwrite(path.join(debug_dir, save_name), canvas)
            except Exception as e:
                print(f"[warn] saving face debug failed: {e}")

        # checkpoint
        if iteration in args.checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            save_data = gaussians.capture()
            save_data['iteration'] = iteration
            torch.save(save_data, path.join(args.out_dir, 'chkpnt' + str(iteration) + '.pth'))

report_cnt = 0
report_data = {}

@torch.no_grad()
def training_report(scene: Scene, gaussians: GaussianModel, iteration, test_iterations, loss_dict, background):
    global report_cnt, report_data
    tb_writer = scene.tb_writer
    report_cnt += 1
    for k, v in loss_dict.items():
        report_data[k] = report_data.get(k, 0) + v
    if report_cnt >= 10:
        for k, v in report_data.items():
            tb_writer.add_scalar(f'train_loss/{k}', v / report_cnt, iteration)
            report_data[k] = 0
        report_cnt = 0

    if iteration in test_iterations:
        torch.cuda.empty_cache()
        l1_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0

        # Optional FID metric (skips gracefully if unavailable)
        fid = None
        try:
            from torchmetrics.image import FrechetInceptionDistance
            fid = FrechetInceptionDistance(feature=2048).cuda()
        except Exception as e:
            print(f"[info] FID unavailable, skipping: {e}")

        rng = random.Random(0)
        write_idxs = [rng.randint(0, len(scene.trainset)-1) for _ in range(10)]

        test_dataloader = DataLoader(
            dataset=scene.testset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        cam_num = 0
        for cam in test_dataloader:
            cam = data_to_cam(cam, non_blocking=False)
            gaussians.smpl_poses = cam['pose']
            gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']
            # 从cam获取expression和从pose中提取jaw_pose、leye_pose、reye_pose
            # pose结构: [global_orient(3) + body_pose(63) + jaw_pose(3) + leye_pose(3) + reye_pose(3) + left_hand_pose(45) + right_hand_pose(45)]
            gaussians.expression = cam['expression']           # expression: 10维，单独在cam中
            gaussians.jaw_pose = gaussians.smpl_poses[66:69]    # jaw_pose: 3维
            gaussians.leye_pose = gaussians.smpl_poses[69:72]   # leye_pose: 3维
            gaussians.reye_pose = gaussians.smpl_poses[72:75]   # reye_pose: 3维

            image, alpha, info = gaussians.render(cam, background=background)
            image = torch.clamp(image, 0, 1)
            image_gt = cam['image']
            image_gt[~cam['mask']] = background

            # L1 / PSNR
            l1_test += l1_loss(image, image_gt).mean().float()
            psnr_test += psnr(image, image_gt).mean().float()

            # SSIM (as metric, not loss)
            try:
                from utils.loss_utils import ssim_loss, lpips_loss
                ssim_val = (1.0 - ssim_loss(image, image_gt)).mean().float()
                ssim_test += ssim_val
            except Exception as e:
                if cam_num == 0:
                    print(f"[info] SSIM unavailable, skipping: {e}")

            # LPIPS
            try:
                lpips_val = lpips_loss(image, image_gt).mean().float()
                lpips_test += lpips_val
            except Exception as e:
                if cam_num == 0:
                    print(f"[info] LPIPS unavailable, skipping: {e}")

            # FID updates expect CHW uint8/float batches
            if fid is not None:
                try:
                    pred_chw = (image.permute(2,0,1)[None] * 255.0).clamp(0,255).byte()
                    gt_chw = (image_gt.permute(2,0,1)[None] * 255.0).clamp(0,255).byte()
                    fid.update(pred_chw, real=False)
                    fid.update(gt_chw, real=True)
                except Exception as e:
                    print(f"[info] FID update failed for a sample: {e}")
            cam_num += 1

            if cam['idx'] in write_idxs:
                frame_id, cam_id = cam['frame_id'], cam['cam_id']
                tb_writer.add_images(f'train_view_{cam_id:02d}_{frame_id:06d}/render', image.permute(2,0,1)[None], global_step=iteration)
                if iteration == test_iterations[0]:
                    tb_writer.add_images(f'train_view_{cam_id:02d}_{frame_id:06d}/ground_truth', image_gt.permute(2,0,1)[None], global_step=iteration)

        psnr_test /= cam_num
        l1_test /= cam_num
        if ssim_test != 0:
            ssim_test /= cam_num
        if lpips_test != 0:
            lpips_test /= cam_num

        # Compute FID if available
        fid_val = None
        if fid is not None:
            try:
                fid_val = fid.compute()
            except Exception as e:
                print(f"[info] FID compute failed: {e}")

        # Console summary
        msg = f"\n[ITER {iteration}] Evaluating train: L1 {l1_test} PSNR {psnr_test}"
        if ssim_test != 0:
            msg += f" SSIM {ssim_test}"
        if lpips_test != 0:
            msg += f" LPIPS {lpips_test}"
        if fid_val is not None:
            msg += f" FID {fid_val}"
        print(msg)

        # TensorBoard
        tb_writer.add_scalar('train/l1_loss', l1_test, iteration)
        tb_writer.add_scalar('train/psnr', psnr_test, iteration)
        if ssim_test != 0:
            tb_writer.add_scalar('train/ssim', ssim_test, iteration)
        if lpips_test != 0:
            tb_writer.add_scalar('train/lpips', lpips_test, iteration)
        if fid_val is not None:
            tb_writer.add_scalar('train/fid', fid_val, iteration)
        tb_writer.add_scalar('total_gaussians', gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=23456)
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir = pargs.data_dir, pargs.out_dir
    args.ip, args.port = pargs.ip, pargs.port
    os.makedirs(args.out_dir, exist_ok = True)

    OmegaConf.save(args, path.join(args.out_dir, 'config.yaml'))

    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    
    safe_state(False, args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args)
