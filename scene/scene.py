
import os
from os import path
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import open3d as o3d

from tensorboardX import SummaryWriter

from utils.config_utils import Config
from utils.smpl_utils import init_smpl, smpl
from utils.general_utils import serialize_to_list, storePly, fetchPly
from utils.graphics_utils import rand_point_on_mesh
from scene.gaussian_model import GaussianModel
from scene.dataset import get_dataset_type, AVRexDataset

class Scene:
    gaussians: GaussianModel
    tb_writer: SummaryWriter
    trainset: AVRexDataset
    testset: AVRexDataset
    trainloader: DataLoader
    scene_scale = None
    def __init__(self, args: Config, gaussians : GaussianModel):
        tb_writer = SummaryWriter(args.out_dir)
        init_smpl(args.smpl_pkl_path)

        # dataset
        frame_ids = np.arange(args.begin_ith_frame, args.begin_ith_frame+args.frame_interval*args.num_train_frame, args.frame_interval).tolist()
        cam_ids = np.array(args.train_cam_ids).tolist()
        image_scaling = args.image_scaling
        DatasetType = get_dataset_type(args.data_dir)
        print(f'Dataset: {DatasetType.__name__}')
        trainset = DatasetType(
            datadir=args.data_dir,
            frame_ids=frame_ids,
            cam_ids=cam_ids,
            background=np.array(args.background),
            image_scaling=image_scaling,
            is_in_memory=args.data_in_memory,
        )
        test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.frame_interval*args.test.num_frame, args.test.frame_interval).tolist()
        test_cam_ids = np.array(args.test.cam_ids).tolist()
        testset = DatasetType(
            datadir=args.data_dir,
            frame_ids=test_frame_ids,
            cam_ids=test_cam_ids,
            background=np.array(args.background),
            image_scaling=image_scaling,
        )
        print(f'Training images: {len(trainset)} Test images: {len(testset)}')

        # dataloader
        trainloader = DataLoader(
            dataset=trainset,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            persistent_workers=False,
            pin_memory=True,
        )

        # collect all the poses and cams from dataset, and dump them to json file
        all_poses, all_Th, all_Rh, all_expression = {}, {}, {}, {}
        cam_list, pose_list = {}, {}
        trainset.is_load_image = False
        beta = trainset[0]['beta']

        for data in trainset:
            cam_id, frame_id = data['cam_id'], data['frame_id']
            if str(frame_id) not in all_poses:
                all_poses[str(frame_id)] = data['pose']
                all_Th[str(frame_id)] = data['Th']
                all_Rh[str(frame_id)] = data['Rh']
                all_expression[str(frame_id)] = data['expression']
            
                pose_list[str(frame_id)] = dict(frame_id=frame_id, pose=data['pose'], Th=data['Th'], Rh=data['Rh'], expression=data['expression'])

            if str(cam_id) not in cam_list:
                cam_list[str(cam_id)] = dict(cam_id=cam_id, w2c=data['w2c'], K=data['K'])
        trainset.is_load_image = True

        cam_list = sorted(cam_list.values(), key=lambda x: x['cam_id'])
        pose_list = sorted(pose_list.values(), key=lambda x: x['frame_id'])
        with open(path.join(args.out_dir, "poses.json"), 'w') as file:
            json.dump(serialize_to_list(pose_list), file)    
        with open(path.join(args.out_dir, "cameras.json"), 'w') as file:
            json.dump(serialize_to_list(cam_list), file)
        
        # skinning weights
        os.makedirs(path.join(args.data_dir, 'gaussian'), exist_ok=True)
        weights_grid_path = path.join(args.data_dir, 'gaussian/lbs_weights_grid.npz')
        if not path.exists(weights_grid_path):
            raise FileNotFoundError(f"权重网格文件不存在: {weights_grid_path}\n"
                                   f"请先运行: python script/gen_weight_volume.py --data_dir {args.data_dir} 来生成权重文件")
        grid_info = dict(np.load(weights_grid_path, allow_pickle=True))
        # Validate LBS grid channels vs SMPL-X joint count
        try:
            grid_shape = grid_info['grid'].shape  # (X, Y, Z, P)
            P_grid = int(grid_shape[-1])
        except Exception:
            raise RuntimeError(f"无法读取 LBS 网格通道数，文件格式异常: {weights_grid_path}")

        P_model = int(len(smpl.model.parents))
        if P_grid != P_model:
            raise RuntimeError(
                f"LBS权重网格关节数({P_grid}) 与 SMPL-X 模型关节数({P_model}) 不匹配。\n"
                f"请重新生成权重文件以包含与当前模型一致的关节集合（含下颌/眼等）：\n"
                f"  python script/gen_weight_volume.py --data_dir {args.data_dir} --smpl_path {args.smpl_pkl_path}\n"
                f"当前权重文件: {weights_grid_path}"
            )
        if P_grid <= 24:
            print(
                f"[警告] 检测到权重通道数为 {P_grid}，很可能不包含面部/手部关节，\n"
                f"嘴部/眼部的几何驱动将受限。建议用 SMPL-X 权重重新生成 lbs_weights_grid.npz。"
            )
        print(P_grid)
        # initialize gaussian model
        scene_scale = trainset.get_scene_scale(args.data_dir) * 1.1
        tpose_model = smpl.model(betas=beta[None], body_pose=smpl.smpl_tpose[None,3*1:22*3])
        # Ensure t_joints count matches joint_parents length exactly (e.g., 55 for SMPL-X)
        num_joints = len(smpl.model.parents)
        t_joints = tpose_model.joints.detach().numpy()[0, :num_joints]
        xyz_path = path.join(args.data_dir, 'gaussian/init_body_points.ply')

        temp_path = path.join(args.data_dir, 'gaussian/template.ply')
        if not path.exists(temp_path):
            print('No template found, using SMPLX mesh')
            bigpose_model = smpl.model(betas=beta[None], body_pose=smpl.smpl_bigpose[None,3*1:22*3])
            verts = bigpose_model.vertices[0].detach().float().numpy()
            faces = smpl.model.faces

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d.io.write_triangle_mesh(temp_path, mesh)

        mesh = o3d.io.read_triangle_mesh(path.join(args.data_dir, 'gaussian/template.ply'))
        verts = np.array(mesh.vertices).astype(np.float32)
        faces = np.array(mesh.triangles).astype(np.float32)

        face_vertices = None
        face_vertex_path = path.abspath(path.join(path.dirname(__file__), '..', 'assets', 'SMPL-X__FLAME_vertex_ids.npy'))
        print(f"[debug] face vertex path: {face_vertex_path}")
        if path.exists(face_vertex_path):
            try:
                face_vertex_ids = np.load(face_vertex_path).astype(np.int64)
                print(f"[debug] loaded vertex id array with shape {face_vertex_ids.shape} and dtype {face_vertex_ids.dtype}")
                valid_mask = (face_vertex_ids >= 0) & (face_vertex_ids < verts.shape[0])
                face_vertex_ids = face_vertex_ids[valid_mask]
                if face_vertex_ids.size > 0:
                    face_vertices = verts[face_vertex_ids]
                    print(f"Loaded {face_vertex_ids.size} face vertices for region initialization")
                else:
                    print("[warn] face vertex list empty after validation")
            except Exception as e:
                print(f"[warn] failed to load face vertex ids from {face_vertex_path}: {e}")
        else:
            print(f"[warn] face vertex definition not found at {face_vertex_path}")

        if path.exists(xyz_path):
            xyz = np.array(fetchPly(xyz_path)[0], dtype=np.float32)
        else:
            print('Initialize Gaussians on Mesh...')
            xyz = rand_point_on_mesh(verts, faces, pts_num=args.init_num_gs)
            storePly(xyz_path, xyz, np.zeros_like(xyz))

        xyz_ft = rand_point_on_mesh(verts, faces, pts_num=args.num_features, init_factor=7)
        xyz_vt = rand_point_on_mesh(verts, faces, pts_num=args.num_verts, init_factor=7)

        gaussians.create_from_pcd(
            xyz=xyz,
            t_joints=t_joints,
            joint_parents=smpl.model.parents,
            lbs_weights_grid_info=grid_info, 
            all_poses=all_poses,
            xyz_ft=xyz_ft,
            xyz_vt=xyz_vt,
            face_vertices=face_vertices,
            data_dir=args.data_dir,
        )

        self.tb_writer = tb_writer
        self.gaussians = gaussians
        self.scene_scale = scene_scale
        self.trainset = trainset
        self.testset = testset
        self.trainloader = trainloader
