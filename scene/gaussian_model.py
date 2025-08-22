import torch
import numpy as np
from torch import nn
import os

from scipy.spatial.transform import Rotation
import torch.nn.functional as F
from torch.func import vmap, functional_call, stack_module_state
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from pytorch3d.ops import knn_points
from gsplat import rasterization, quat_scale_to_covar_preci, spherical_harmonics
from utils.sss_sghmc import AdamSGHMC

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.
    Same as in talkbody4D.py
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    axis = axis_angle / (angles + 1e-6)  # Avoid division by zero
    x, y, z = torch.unbind(axis, dim=-1)

    sin_theta = torch.sin(angles)
    cos_theta = torch.cos(angles)
    one_minus_cos_theta = 1 - cos_theta

    o = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([o, -z, y], dim=-1),
            torch.stack([z, o, -x], dim=-1),
            torch.stack([-y, x, o], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    R = (
        eye
        + sin_theta.unsqueeze(-1) * K
        + one_minus_cos_theta.unsqueeze(-1) * torch.matmul(K, K)
    )

    return R

from scene.mlp import MLP, vmap_mlp
from utils.smpl_utils import smpl, interpolate_skinningfield, rigid_transform_tensor, rigid_transform_numba
from utils.config_utils import Config
from utils.sh_utils import RGB2SH
from utils.sss_reloc import compute_relocation_student_t_cuda

class GaussianModel:

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = F.normalize

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = torch.logit

        # SSS: degree of freedom activation (clamped to [1, 10000])
        from torch import nn as _nn
        self.nu_degree_activation = _nn.Hardtanh(1, 10000)

    def __init__(self):

        self._xyz = torch.empty(0)
        self.xyz_offset = torch.empty(0)
        self.dxyz_vt = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._shN = torch.empty(0)
        self.sh_degree = 0

        self.xyz_vt = torch.empty(0)
        self.xyz_ft = torch.empty(0)

        # basis property definition
        self.num_vt_basis = 15     # Control point basis number
        self.num_basis = 15        # Gaussian property basis number

        self.encoder_feat_params = None
        self.encoder_feat_model_meta = None

        self.dxyz_bs = torch.empty(0)
        self.sh0_bs = torch.empty(0)
        self.shN_bs = torch.empty(0)
        self.scaling_bs = torch.empty(0)
        self.rotation_bs = torch.empty(0)
        self.opacity_bs = torch.empty(0)

        # lbs weights
        self._weights = None

        # pose
        self._Rh = torch.empty(0)
        self._Th = torch.empty(0)
        self.Ac_inv = torch.empty(0)
        self._smpl_poses = torch.empty(0)
        self.smpl_poses_cuda = torch.empty(0)
        self.t_joints = torch.empty(0)
        self.joint_parents = torch.empty(0)

        self.all_poses = torch.empty(0)
        
        # facial parameters
        self.expression = torch.zeros(10, dtype=torch.float32)
        self.jaw_pose = torch.zeros(3, dtype=torch.float32)
        self.leye_pose = torch.zeros(3, dtype=torch.float32)
        self.reye_pose = torch.zeros(3, dtype=torch.float32)

        # cache
        self.cache_dict = {}

        # optimizer
        self.optimizers = None
        self.schedulers = None
        self.sss_optimizer = None

        # SSS additions
        self._degree = torch.empty(0)
        self._negative = torch.empty(0)
        # Render backend: 'gsplat' (default) or 'sss'
        self.renderer_backend = 'gsplat'
        # knn
        self.nbr_gs = torch.empty(0)
        self.nbr_gs_invdist = torch.empty(0)
        self.nbr_vt = torch.empty(0)
        self.nbr_gsft = torch.empty(0)
        self.nbr_vtft = torch.empty(0)
        self.nbr_gsft_wght = torch.empty(0)
        self.nbr_vtft_wght = torch.empty(0)

        # misc
        self.scene_scale = None
        self.is_dxyz_bs = False     # whether to use control point basis
        self.is_gsparam_bs = False  # whether to use Gaussian property basis

        self.is_test = False        # whether to use PCA 

        self.setup_functions()

    def capture(self):
        data = {
            '_xyz': self._xyz,
            'xyz_offset': self.xyz_offset,
            'dxyz_vt': self.dxyz_vt,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_opacity': self._opacity,
            '_sh0': self._sh0,
            '_shN': self._shN,
            'sh_degree': self.sh_degree,

            '_weights': self.get_weights,

            't_joints': self.t_joints,
            'all_poses': self.all_poses,
            'joint_parents': self.joint_parents,

            'nbr_gs_invdist': self.nbr_gs_invdist,
            'nbr_gs': self.nbr_gs,
            'nbr_vt': self.nbr_vt,
            'nbr_gsft': self.nbr_gsft,
            'nbr_vtft': self.nbr_vtft,
            'nbr_gsft_wght': self.nbr_gsft_wght,
            'nbr_vtft_wght': self.nbr_vtft_wght,

            'xyz_vt': self.xyz_vt,
            'xyz_ft': self.xyz_ft,

            'num_vt_basis': self.num_vt_basis,
            'num_basis': self.num_basis,

            'encoder_feat_params': self.encoder_feat_params,
            'encoder_feat_model_meta': self.encoder_feat_model_meta,

            'dxyz_bs': self.dxyz_bs,
            'sh0_bs': self.sh0_bs,
            'shN_bs': self.shN_bs,
            'scaling_bs': self.scaling_bs,
            'rotation_bs': self.rotation_bs,
            'opacity_bs': self.opacity_bs,

            'is_dxyz_bs': self.is_dxyz_bs,
            'is_gsparam_bs': self.is_gsparam_bs,

            # SSS
            '_degree': self._degree,
            '_negative': self._negative,
        }
        return data
    
    def restore(self, data):
        def loader(s):
            if s in data: return data[s]
            else: print(f'NO DATA {s}!')
            return None

        self._xyz = data['_xyz']
        self.xyz_offset = data['xyz_offset']
        self.dxyz_vt = data['dxyz_vt']
        self._opacity = data['_opacity']
        self._rotation = data['_rotation']
        self._scaling = data['_scaling']
        self._sh0 = data['_sh0']
        self._shN = loader('_shN')
        self.sh_degree = data['sh_degree']

        self._weights = data['_weights']

        self.t_joints = loader('t_joints')
        self.all_poses = loader('all_poses')
        self.joint_parents = loader('joint_parents')

        self.nbr_gs = loader('nbr_gs')
        self.nbr_vt = loader('nbr_vt')
        self.nbr_gs_invdist = loader('nbr_gs_invdist')
        self.nbr_gsft = loader('nbr_gsft')
        self.nbr_vtft = loader('nbr_vtft')
        self.nbr_gsft_wght = loader('nbr_gsft_wght')
        self.nbr_vtft_wght = loader('nbr_vtft_wght')

        self.xyz_vt = loader('xyz_vt')
        self.xyz_ft = loader('xyz_ft')

        self.num_vt_basis = loader('num_vt_basis')
        self.num_basis = loader('num_basis')

        self.encoder_feat_params = loader('encoder_feat_params')
        self.encoder_feat_model_meta = loader('encoder_feat_model_meta')

        self.dxyz_bs = loader('dxyz_bs')
        self.sh0_bs = loader('sh0_bs')
        self.shN_bs = loader('shN_bs')
        self.scaling_bs = loader('scaling_bs')
        self.rotation_bs = loader('rotation_bs') 
        self.opacity_bs = loader('opacity_bs')

        self.is_dxyz_bs = loader('is_dxyz_bs')
        self.is_gsparam_bs = loader('is_gsparam_bs')

        self.init()

    def init(self):
        self.init_body() 
        self.reset_pose()   

    @property
    def get_cano_scaling(self):
        if 'get_cano_scaling' in self.cache_dict: return self.cache_dict['get_cano_scaling'] 
        if not self.is_gsparam_bs: 
            scaling = self.scaling_activation(self._scaling)
        else:
            features = self.get_encoded_feature_gsparam_weight
            dscaling = torch.einsum('nc,ncl->nl', features, self.scaling_bs)

            scaling = self._scaling + dscaling
            scaling = self.scaling_activation(scaling)
        
        self.cache_dict['get_cano_scaling'] = scaling
        return scaling
    
    @property
    def get_weights(self):
        if self._weights is None:
            xyz = self._xyz
            weights = interpolate_skinningfield(self.weights_grid_info, xyz)
            self._weights = weights
        else:
            weights = self._weights
        return weights

    @property
    def get_rigid_transform(self):
        if 'get_rigid_transform' in self.cache_dict: return self.cache_dict['get_rigid_transform']
        pose = self.smpl_poses.cpu().numpy()
        joints = self.t_joints.cpu().numpy()
        parent = self.joint_parents.cpu().numpy()
        Ac_inv = self.Ac_inv.cpu().numpy()

        # Sanity checks to ensure pose/joints/parents alignment (e.g., 55 for SMPL-X)
        nj_pose = pose.reshape(-1, 3).shape[0]
        nj_joint = joints.shape[0]
        nj_parent = len(parent)
        assert nj_pose == nj_parent, f"smpl_poses joints ({nj_pose}) != parents ({nj_parent})"
        assert nj_joint == nj_parent, f"t_joints ({nj_joint}) != parents ({nj_parent})"
        assert Ac_inv.shape[0] == nj_parent, f"Ac_inv joints ({Ac_inv.shape[0]}) != parents ({nj_parent})"

        rots = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().astype(np.float32)
        A = rigid_transform_numba(rots, joints, parent)
        G = np.matmul(A, Ac_inv)

        data = [torch.as_tensor(d).cuda(non_blocking=True) for d in [rots, G]]
        self.cache_dict['get_rigid_transform'] = data
        return data

    @property
    def get_Gweights(self):
        if 'get_Gweights' in self.cache_dict: return self.cache_dict['get_Gweights']

        # Rots = batch_rodrigues(self.smpl_poses.reshape(-1,3))
        # A = batch_rigid_transform(Rots[None], self.t_joints[None], self.joint_parents)[1][0]
        # G = torch.matmul(A, self.Ac_inv)
        
        G = self.get_rigid_transform[1]
        G_weight = torch.einsum('vp,pij->vij', self.get_weights, G)

        self.cache_dict['get_Gweights'] = G_weight
        return G_weight

    @property
    def get_cano_rotation(self):
        if not self.is_gsparam_bs: 
            rotation = self.rotation_activation(self._rotation)
        else:
            features = self.get_encoded_feature_gsparam_weight
            drotation = torch.einsum('nc,ncl->nl', features, self.rotation_bs)

            rotation = self._rotation + drotation
            rotation = self.rotation_activation(rotation)

        return rotation

    def get_covariance(self, scaling_modifier=1):
        rots = self.get_Gweights[:,:3,:3].contiguous()
        covs = quat_scale_to_covar_preci(
            quats=self.get_cano_rotation,
            scales=self.get_cano_scaling * scaling_modifier,
            compute_preci=False,
        )[0]

        if self.Rh is not None: rots = self.Rh @ rots
        covs = rots @ covs @ rots.transpose(-1,-2)
        return covs

    @property
    def get_joint_features(self):

        if self.is_test:
            sigma_pca = 2.0
            features = self.smpl_poses_cuda[1*3:22*3][None]
            lowdim_pose_conds = self.pca.transform(features)
            std = self.pca_std
            lowdim_pose_conds = torch.maximum(lowdim_pose_conds, -sigma_pca * std)
            lowdim_pose_conds = torch.minimum(lowdim_pose_conds, sigma_pca * std)
            body_features = self.pca.inverse_transform(lowdim_pose_conds).reshape(-1)
        else:
            body_features = self.smpl_poses_cuda[3:3*22]  # 63维 body poses

        # 拼接表情和下颌参数
        expression_cuda = self.expression.cuda()  # 10维
        jaw_pose_cuda = self.jaw_pose.cuda()      # 3维
        leye_pose_cuda = self.leye_pose.cuda()    # 3维
        reye_pose_cuda = self.reye_pose.cuda()    # 3维
        
        # 组合成82维特征: body(63) + expression(10) + jaw(3) + leye_pose(3) + reye_pose(3)
        features = torch.cat([body_features, expression_cuda, jaw_pose_cuda, leye_pose_cuda, reye_pose_cuda])
        
        # Debug: Print feature dimensions
        if features.shape[0] != 82:
            print(f"Warning: Expected 82 features, got {features.shape[0]}")
            print(f"body_features: {body_features.shape[0]}, expression: {expression_cuda.shape[0]}, jaw_pose: {jaw_pose_cuda.shape[0]}, leye_pose: {leye_pose_cuda.shape[0]}, reye_pose: {reye_pose_cuda.shape[0]}")
            # Truncate to 82 if needed
            features = features[:82]

        return features

    @torch.no_grad()
    def prepare_test(self):
        pose_set = []
        for k, v in self.all_poses.items():
            pose_set.append(v[1*3:22*3].detach())     
        N_pose = len(pose_set)
        pose_set = torch.stack(pose_set, dim=0).reshape(N_pose,21,3).cpu().numpy()
        features = pose_set.reshape(N_pose, -1)

        pca_num = 20

        features = torch.as_tensor(features).cuda()
        from torch_pca import PCA
        self.pca = PCA(n_components=pca_num)
        self.pca.fit(features)
        self.pca_std = torch.sqrt(self.pca.explained_variance_)

        print(f'Use PCA components: {pca_num}')

    @property
    def get_encoded_feature(self):
        if 'get_encoded_feature' in self.cache_dict: return self.cache_dict['get_encoded_feature']
        features = self.get_joint_features
        N_feat = len(self.encoder_feat_params['layers.0.weight'])
        features = features.tile([N_feat, 1])
        features = vmap_mlp(self.encoder_feat_params, features)

        self.cache_dict['get_encoded_feature'] = features
        return features

    @property
    def get_encoded_feature_gsparam_weight(self):
        if 'get_encoded_feature_gsparam_weight' in self.cache_dict: return self.cache_dict['get_encoded_feature_gsparam_weight']
        features = self.get_encoded_feature[...,:self.num_basis]
        features = torch.einsum('nrc,nr->nc', features[self.nbr_gsft], self.nbr_gsft_wght)

        self.cache_dict['get_encoded_feature_gsparam_weight'] = features
        return features

    @property
    def get_dxyz_vt(self):
        if 'get_dxyz_vt' in self.cache_dict: return self.cache_dict['get_dxyz_vt']
        if not self.is_dxyz_bs: return self.dxyz_vt

        features = self.get_encoded_feature[...,self.num_basis:]
        features = torch.einsum('nrc,nr->nc', features[self.nbr_vtft], self.nbr_vtft_wght)

        dxyz_vt = torch.einsum('vc,vcl->vl', features, self.dxyz_bs)

        dxyz_vt = self.dxyz_vt + dxyz_vt
        self.cache_dict['get_dxyz_vt'] = dxyz_vt

        return dxyz_vt

    @property
    def get_dxyz(self):
        if 'get_dxyz' in self.cache_dict: return self.cache_dict['get_dxyz']

        dxyz = torch.sum(self.nbr_gs_invdist[...,None] * self.get_dxyz_vt[self.nbr_gs], dim=1) / torch.sum(self.nbr_gs_invdist, dim=-1)[...,None]
        self.cache_dict['get_dxyz'] = dxyz
        return dxyz
    
    @property
    def get_cano_xyz(self):
        if 'get_cano_xyz' in self.cache_dict: return self.cache_dict['get_cano_xyz']
        xyz = self._xyz + self.get_dxyz + torch.tanh(self.xyz_offset) * 0.008   # A trick to allow Gaussians to move freely within a small range
        self.cache_dict['get_cano_xyz'] = xyz
        return xyz

    @property
    def get_xyz(self):
        if 'get_xyz' in self.cache_dict: return self.cache_dict['get_xyz']
        
        # Step 1: 获取canonical坐标（SMPL/SMPLX在T-pose或zero-pose下的坐标）
        xyz = self.get_cano_xyz
        
        # Step 2: 应用Linear Blend Skinning (LBS)变换
        # 根据姿态参数（smpl_poses）和关节权重进行骨骼绑定变换
        xyz = torch.einsum('vij,vj->vi', self.get_Gweights, F.pad(xyz,(0,1),value=1))[:,:3]
        
        # Step 3 & 4: 应用全局变换（SQ02数据集的核心）
        # 在SQ02格式中：
        # - SMPLX模型使用 global_orient=0, transl=0 生成vertices
        # - 真实的全局变换保存在Rh（旋转）和Th（平移）参数中
        # - 变换顺序：先旋转再平移，与talkbody4D.py保持一致
        if self.Rh is not None: 
            # 使用右乘：xyz @ Rh.T，与talkbody4D.py中的 verts @ Rh.transpose(1, 2) 一致
            xyz = torch.einsum('vi,ji->vj', xyz, self.Rh)  # 等价于 xyz @ Rh.T
        xyz = xyz + self.Th

        self.cache_dict['get_xyz'] = xyz
        return xyz

    @property
    def get_opacity(self):
        if not self.is_gsparam_bs:
            opacity = self.opacity_activation(self._opacity)     
        else:
            features = self.get_encoded_feature_gsparam_weight
            dopacity = torch.einsum('nc,nc->n', features, self.opacity_bs)

            opacity = self._opacity + dopacity
            opacity = self.opacity_activation(opacity)

        return opacity

    @property
    def get_sh(self):
        if 'get_sh' in self.cache_dict: return self.cache_dict['get_sh']

        if self.sh_degree == 0: 
            sh = self._sh0
        else:
            sh = torch.cat([self._sh0, self._shN], dim=1)

        if self.is_gsparam_bs:

            features = self.get_encoded_feature_gsparam_weight
            dsh0 = torch.einsum('nc,ncxy->nxy', features, self.sh0_bs)
            if self.sh_degree == 0: 
                dsh = dsh0
            else: 
                dshN = torch.einsum('nc,ncxy->nxy', features, self.shN_bs)
                dsh = torch.cat([dsh0, dshN], dim=1)

            sh = sh + dsh

        self.cache_dict['get_sh'] = sh
        return sh

    def get_color(self, cam_pos):
        if 'get_color' in self.cache_dict: return self.cache_dict['get_color']

        if self.sh_degree > 0:
            rots = self.get_Gweights[:,:3,:3]
            # with torch.set_grad_enabled(False):
            #     rots = polar_decomposition_newton_schulz(rots)

            dirs = F.normalize(cam_pos - self.get_xyz, dim=-1)
            invrots = rots.transpose(-1,-2)
            dirs = torch.einsum('nij,nj->ni',invrots, dirs)
        else:
            dirs = torch.ones_like(self._xyz)

        sh = self.get_sh
        color = spherical_harmonics(self.sh_degree, dirs, sh)
        color = torch.clamp_min(color + 0.5, 0)

        self.cache_dict['get_color'] = color

        return color

    def create_from_pcd(self, xyz=None, t_joints=None, joint_parents=None, all_poses=None, lbs_weights_grid_info=None, xyz_vt=None, xyz_ft=None):
        xyz = torch.as_tensor(xyz).float().cuda() # [N,3]
        N = xyz.shape[0]
        print("Number of points at initialization : ", N)

        init_opacity = 0.8
        init_color = 0.5

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = knn_points(xyz[None], xyz[None], K=4)[0][0,:,1:].mean(dim=-1, keepdim=True)  
        scale = self.scaling_inverse_activation(torch.sqrt(dist2_avg)).tile([1,3])  # [N,3]
        rotation = torch.zeros((N, 4)).float().cuda()
        rotation[:, 0] = 1  # [N,4]
        opacity = torch.full((N,), self.inverse_opacity_activation(torch.tensor(init_opacity))).float().cuda()  # [N,]
        sh0 = torch.full((N, 1, 3), RGB2SH(init_color)).float().cuda() 
        shN = torch.zeros((N, 3, 3)).float().cuda()
        xyz_offset = torch.zeros_like(xyz)

        self._xyz = xyz
        self.xyz_offset = nn.Parameter(xyz_offset.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._sh0 = nn.Parameter(sh0.requires_grad_(True))
        self._shN = nn.Parameter(shN.requires_grad_(True))

        # SSS: initialize degree (nu) and negative (learnable as requested)
        nu_init = 10.0
        negative_init = 1.0
        degree = torch.full((N, 1), float(nu_init), dtype=torch.float32, device=xyz.device)
        negative = torch.full((N, 1), float(negative_init), dtype=torch.float32, device=xyz.device)
        self._degree = nn.Parameter(degree.requires_grad_(True))
        self._negative = nn.Parameter(negative.requires_grad_(True))

        self.t_joints = torch.as_tensor(t_joints).detach().float().cpu()
        self.joint_parents = torch.as_tensor(joint_parents).detach().cpu()

        for key in all_poses: all_poses[key] = torch.as_tensor(all_poses[key]).float().cpu()
        self.all_poses = all_poses

        ginfo = lbs_weights_grid_info
        for key in ['grid', 'bbox_min', 'bbox_max', 'grid_dims']: ginfo[key] = torch.as_tensor(ginfo[key]).detach().cuda()
        self.weights_grid_info = ginfo

        # Pose encoder - 扩展输入维度支持表情: body(63) + expression(10) + jaw(3) + leye_pose(3) + reye_pose(3) = 82
        models = [MLP(layers_size_list=[82, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]) for i in range(len(xyz_ft))]
        params, _ = stack_module_state(models)
        self.encoder_feat_model_meta = MLP(layers_size_list=[82, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]).to('meta')
        for k, v in params.items():
            params[k] = nn.Parameter(v.cuda().requires_grad_(True))
        self.encoder_feat_params = params

        # basis
        dxyz_bs = torch.zeros((len(xyz_vt), self.num_vt_basis, 3)).float().cuda()
        sh0_bs = torch.zeros((N, self.num_basis, 1, 3)).float().cuda()
        shN_bs = torch.zeros((N, self.num_basis, 3, 3)).float().cuda()
        scaling_bs = torch.zeros((N, self.num_basis, 3)).float().cuda()
        rotation_bs = torch.zeros((N, self.num_basis, 4)).float().cuda()
        opacity_bs = torch.zeros((N, self.num_basis)).float().cuda()
        for data in [dxyz_bs, sh0_bs, scaling_bs, rotation_bs, opacity_bs]:
            nn.init.uniform_(data[0], -0.002, 0.002)
            data[1:] = data[0]
        self.dxyz_bs = nn.Parameter(dxyz_bs.requires_grad_(True))
        self.sh0_bs = nn.Parameter(sh0_bs.requires_grad_(True))
        self.shN_bs = nn.Parameter(shN_bs.requires_grad_(True))
        self.scaling_bs = nn.Parameter(scaling_bs.requires_grad_(True))
        self.rotation_bs = nn.Parameter(rotation_bs.requires_grad_(True))
        self.opacity_bs = nn.Parameter(opacity_bs.requires_grad_(True))

        xyz_ft = torch.as_tensor(xyz_ft).float().cuda()
        xyz_vt = torch.as_tensor(xyz_vt).float().cuda()
        self.dxyz_vt = nn.Parameter(torch.zeros_like(xyz_vt).float().cuda().requires_grad_(True))

        self.prepare_interpolating_weights(xyz_ft, xyz_vt)

        self.init()

    def training_setup(self, args: Config, scene_scale):
        use_sghmc = getattr(args, 'optimizer', 'adam') == 'sghmc'
        self.optimizers, self.schedulers = None, None
        if use_sghmc:
            # Full SSS-style optimizer: one AdamSGHMC manages all groups; 'xyz' uses SGHMC, others get Adam-like updates
            C_burnin = getattr(args, 'C_burnin', 5e3)
            C = getattr(args, 'C', 1.3e2)
            burnin_iterations = getattr(args, 'burnin_iterations', 7000)
            param_groups = [
                {'params': [self.xyz_offset], 'lr': args.position_lr * scene_scale, 'name': 'xyz', 'mdecay': C, 'mdecay_burnin': C_burnin, 'burnin_iterations': burnin_iterations, 'scale_grad': 1.0},
                {'params': [self._scaling], 'lr': args.scaling_lr, 'name': 'scaling'},
                {'params': [self._rotation], 'lr': args.rotation_lr, 'name': 'rotation'},
                {'params': [self._opacity], 'lr': args.opacity_lr, 'name': 'opacity'},
                {'params': [self._sh0], 'lr': args.color_lr, 'name': 'f_dc'},
                {'params': [self._shN], 'lr': args.color_lr / 20.0, 'name': 'f_rest'},
                {'params': [self._degree], 'lr': getattr(args, 'degree_lr', 5e-4), 'name': 'degree'},
                {'params': [self._negative], 'lr': getattr(args, 'negative_lr', 1e-4), 'name': 'negative'},
                {'params': [self.dxyz_vt], 'lr': args.position_lr * scene_scale / 10.0, 'name': 'dxyz_vt'},
                {'params': [self.scaling_bs], 'lr': args.scaling_lr / 5.0, 'name': 'dscales_bs'},
                {'params': [self.rotation_bs], 'lr': args.rotation_lr / 5.0, 'name': 'dquats_bs'},
                {'params': [self.opacity_bs], 'lr': args.opacity_lr / 5.0, 'name': 'dopacities_bs'},
                {'params': [self.sh0_bs], 'lr': args.color_lr / 5.0, 'name': 'dsh0_bs'},
                {'params': [self.shN_bs], 'lr': args.color_lr / 200.0, 'name': 'dshN_bs'},
            ]
            # Encoder params as one group
            if self.encoder_feat_params is not None:
                param_groups.append({'params': list(self.encoder_feat_params.values()), 'lr': args.encoder_lr, 'name': 'encoder'})
            self.sss_optimizer = AdamSGHMC(params=param_groups, eps=1e-15, scale_grad=1.0)
        else:
            # Fallback to original Adam + schedulers (kept for compatibility)
            eps=1e-15 
            betas = (1 - 1 * (1 - 0.9), 1 - 1 * (1 - 0.999))
            decay = 0.001
            optimizers = {
                'dxyz': Adam([self.dxyz_vt], args.position_lr * scene_scale, betas, eps),
                'scales': Adam([self._scaling], args.scaling_lr, betas, eps),
                'quats': Adam([self._rotation], args.rotation_lr, betas, eps),
                'opacities': Adam([self._opacity], args.opacity_lr, betas, eps),
                'sh0': Adam([self._sh0], args.color_lr, betas, eps),
                'shN': Adam([self._shN], args.color_lr / 20, betas, eps),
                'dxyz_bs': Adam([self.dxyz_bs], args.position_lr * scene_scale / 10, betas, eps),
                'dscales_bs': Adam([self.scaling_bs], args.scaling_lr / 5, betas, eps),
                'dquats_bs': Adam([self.rotation_bs], args.rotation_lr / 5, betas, eps),
                'dopacities_bs': Adam([self.opacity_bs], args.opacity_lr / 5, betas, eps),
                'dsh0_bs': Adam([self.sh0_bs], args.color_lr / 5, betas, eps),
                'dshN_bs': Adam([self.shN_bs], args.color_lr / 200, betas, eps),
                'encoder_feat_params': AdamW(self.encoder_feat_params.values(), args.encoder_lr, betas, eps, decay),
                'xyz_offset': Adam([self.xyz_offset], args.xyz_offset_lr, betas, eps),
                'degree': Adam([self._degree], getattr(args, 'degree_lr', 5e-4), betas, eps),
                'negative': Adam([self._negative], getattr(args, 'negative_lr', 1e-4), betas, eps),
            }
            schedulers = [
                ExponentialLR(optimizers['dxyz'], gamma=0.01 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['scales'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['quats'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['opacities'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['sh0'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['shN'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['dxyz_bs'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['encoder_feat_params'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['xyz_offset'], gamma=0.1 ** (1.0 / args.iterations)),
            ]
            self.optimizers = optimizers
            self.schedulers = schedulers

    def optimizer_step(self):
        # If using SSS optimizer, the step is handled in train loop (needs sig & cov). No-op here.
        if self.sss_optimizer is not None:
            self.cache_dict = {}
            return
        if self.optimizers is not None:
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if self.schedulers is not None:
            for scheduler in self.schedulers:
                scheduler.step()
        self.cache_dict = {}

    def render(self, cam, override_color=None, scaling_modifier=1.0, background=None):
        if self.renderer_backend == 'sss':
            from render.sss_adapter import render_sss
            image, alpha, info = render_sss(cam, self, background, compute_cov3D_python=True)
            # Adapter returns full image tensor; keep interface consistent
            return image, alpha, info
        else:
            sh = self.get_sh      # can be faster
            covars = self.get_covariance(scaling_modifier)
            if override_color is None:
                cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3,3]
                override_color = self.get_color(cam_pos)
            
            image, alpha, info = rasterization(
                means=self.get_xyz,
                quats=None,
                scales=None,
                opacities=self.get_opacity,
                colors=override_color,
                viewmats=cam['w2c'][None],  # [1, 4, 4]
                Ks=cam['K'][None],  # [1, 3, 3]
                width=cam['width'],
                height=cam['height'],
                packed=False,
                near_plane=0.1,
                backgrounds=background[None],  # [1, 3]
                covars=covars,
            )
            return image[0], alpha[0], info

    def init_body(self):
        # Rots = batch_rodrigues(smpl.smpl_bigpose.reshape(-1,3)).cuda()
        # Ac = batch_rigid_transform(Rots[None], self.t_joints[None], self.joint_parents)[1][0]
        Ac = rigid_transform_tensor(smpl.smpl_bigpose, self.t_joints, self.joint_parents).cpu()
        self.Ac_inv = torch.linalg.inv(Ac)
        self.reset_pose()

    def reset_pose(self):
        self.Rh = torch.eye(3, dtype=torch.float32, device='cpu')
        self.Th = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.smpl_poses = smpl.smpl_tpose.cpu()

    @property
    def smpl_poses(self):
        return self._smpl_poses
    
    @smpl_poses.setter
    def smpl_poses(self, value):
        self.cache_dict = {}
        self._smpl_poses = value.cpu()
        self.smpl_poses_cuda = value.cuda(non_blocking=True)

    @property
    def Rh(self):
        return self._Rh
    
    @Rh.setter
    def Rh(self, value):
        self.cache_dict = {}  # 清空缓存，因为Rh改变了
        
        # 检查value的形状来判断是axis-angle还是旋转矩阵
        if value.shape[-1] == 3 and len(value.shape) == 1:
            # axis-angle格式 (3,) - SQ02数据集格式
            if torch.allclose(value, torch.zeros_like(value), atol=1e-5):
                self._Rh = None
            else:
                # 转换axis-angle为旋转矩阵
                rotation_matrix = axis_angle_to_matrix(value.unsqueeze(0)).squeeze(0)  # (3, 3)
                self._Rh = rotation_matrix.cuda(non_blocking=True)
        elif value.shape == (3, 3):
            # 旋转矩阵格式 (3, 3) - 其他数据集格式
            if torch.allclose(value, torch.eye(3, device=value.device), atol=1e-5):
                self._Rh = None
            else:
                self._Rh = value.cuda(non_blocking=True)
        else:
            raise ValueError(f"Rh must be either axis-angle (3,) or rotation matrix (3, 3), got shape {value.shape}")

    @property
    def Th(self):
        return self._Th
    
    @Th.setter
    def Th(self, value):
        self.cache_dict = {}  # 清空缓存，因为Th改变了
        self._Th = value.cuda(non_blocking=True)

    @property
    def get_nu_degree(self):
        # Map raw degree to [1, 10000]
        return self.nu_degree_activation(self._degree)

    @property
    def get_negative(self):
        return self._negative

    def prepare_interpolating_weights(self, xyz_ft, xyz_vt):
        self.xyz_vt = xyz_vt
        self.xyz_ft = xyz_ft

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_vt[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)

        _, idxs, _ = knn_points(
            p1=xyz_vt[None],
            p2=xyz_vt[None],
            K=7,
        )
        nbr_vt = idxs[0]

        self.nbr_gs = nbr_gs
        self.nbr_gs_invdist = nbr_gs_invdist
        self.nbr_vt = nbr_vt

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_gsft = nbr_gs
        self.nbr_gsft_wght = nbr_gs_wght

        dists, idxs, _ = knn_points(
            p1=self.xyz_vt[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_vtft = nbr_gs
        self.nbr_vtft_wght = nbr_gs_wght

    def _update_interpolating_weights_for_new(self, new_xyz):
        from pytorch3d.ops import knn_points as _knn
        xyz_vt = self.xyz_vt
        xyz_ft = self.xyz_ft
        dists, idxs, _ = _knn(p1=new_xyz[None], p2=xyz_vt[None], K=3)
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        self.nbr_gs = torch.cat([self.nbr_gs, nbr_gs], dim=0)
        self.nbr_gs_invdist = torch.cat([self.nbr_gs_invdist, nbr_gs_invdist], dim=0)
        dists, idxs, _ = _knn(p1=new_xyz[None], p2=xyz_ft[None], K=3)
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_gsft = torch.cat([self.nbr_gsft, nbr_gs], dim=0)
        self.nbr_gsft_wght = torch.cat([self.nbr_gsft_wght, nbr_gs_wght], dim=0)

    # =============== SSS densification and recycling ===============
    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs.abs() / (probs.abs().sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs, minlength=self._opacity.shape[0]).unsqueeze(-1)
        return sampled_idxs, ratio

    def _update_params(self, idxs, ratio):
        op = self.get_opacity
        if op.ndim == 2:
            op = op[:, 0]
        neg = self.get_negative
        if neg.ndim == 2:
            neg = neg[:, 0]
        nu = self.get_nu_degree
        nu_flat = nu[:, 0] if nu.ndim == 2 else nu
        new_opacity, new_scaling = compute_relocation_student_t_cuda(
            opacity_old=(op[idxs] * neg[idxs]).contiguous(),
            scale_old=self.get_cano_scaling[idxs].contiguous(),
            nu_degree=nu_flat[idxs].contiguous(),
            N=ratio[idxs, 0].contiguous() + 1,
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=-1.0 + torch.finfo(torch.float32).eps)
        new_opacity = torch.where((new_opacity >= 0) & (new_opacity < 0.005), 0.005, new_opacity)
        new_opacity = torch.where((new_opacity < 0) & (new_opacity > -0.005), -0.005, new_opacity)
        new_opacity = (new_opacity / self.get_negative[idxs]).squeeze(-1)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        new_xyz = self._xyz[idxs]
        new_sh0 = self._sh0[idxs]
        new_shN = self._shN[idxs]
        new_rot = self._rotation[idxs]
        new_deg = self._degree[idxs]
        new_neg = self._negative[idxs]
        return new_xyz, new_sh0, new_shN, new_opacity, new_scaling, new_rot, new_deg, new_neg

    def cat_tensors_to_optimizer(self, tensors_dict, inds):
        if self.sss_optimizer is None:
            return {}
        optimizable_tensors = {}
        for group in self.sss_optimizer.param_groups:
            name = group.get('name', '')
            if name not in tensors_dict:
                continue
            ext = tensors_dict[name]
            param = group['params'][0]
            state = self.sss_optimizer.state.get(param, None)
            new_param = nn.Parameter(torch.cat((param.data, ext.detach()), dim=0).requires_grad_(True))
            if state is not None:
                new_state = {}
                for k, v in state.items():
                    if torch.is_tensor(v):
                        pad = torch.zeros_like(ext.detach())
                        new_state[k] = torch.cat((v, pad), dim=0)
                    else:
                        new_state[k] = v
                del self.sss_optimizer.state[param]
                group['params'][0] = new_param
                self.sss_optimizer.state[new_param] = new_state
            else:
                group['params'][0] = new_param
            optimizable_tensors[name] = new_param
        if 'xyz' in optimizable_tensors:
            self.xyz_offset = optimizable_tensors['xyz']
        if 'scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['scaling']
        if 'rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['rotation']
        if 'opacity' in optimizable_tensors:
            self._opacity = optimizable_tensors['opacity']
        if 'f_dc' in optimizable_tensors:
            self._sh0 = optimizable_tensors['f_dc']
        if 'f_rest' in optimizable_tensors:
            self._shN = optimizable_tensors['f_rest']
        if 'degree' in optimizable_tensors:
            self._degree = optimizable_tensors['degree']
        if 'negative' in optimizable_tensors:
            self._negative = optimizable_tensors['negative']
        return optimizable_tensors

    def replace_tensors_to_optimizer(self, inds=None):
        if self.sss_optimizer is None:
            return
        for group in self.sss_optimizer.param_groups:
            param = group['params'][0]
            state = self.sss_optimizer.state.get(param, None)
            if state is not None and inds is not None and 'momentum' in state:
                try:
                    state['momentum'][inds] = 0
                except Exception:
                    pass

    def replace_tensors_to_optimizer_momentum(self, inds=None):
        if self.sss_optimizer is None or inds is None:
            return
        for group in self.sss_optimizer.param_groups:
            param = group['params'][0]
            state = self.sss_optimizer.state.get(param, None)
            if state is not None and 'momentum' in state:
                try:
                    state['momentum'][inds] = 0
                except Exception:
                    pass

    def densification_postfix(self, new_xyz, new_sh0, new_shN, new_opacity, new_scaling, new_rotation, new_degree, new_negative, indices, reset_params=True):
        self._xyz = torch.cat([self._xyz, new_xyz.detach()], dim=0)
        ext = {
            'xyz': torch.zeros((new_xyz.shape[0],) + self.xyz_offset.shape[1:], device=self.xyz_offset.device, dtype=self.xyz_offset.dtype),
            'f_dc': new_sh0.detach(),
            'f_rest': new_shN.detach(),
            'opacity': new_opacity.detach(),
            'scaling': new_scaling.detach(),
            'rotation': new_rotation.detach(),
            'degree': new_degree.detach(),
            'negative': new_negative.detach(),
        }
        self.cat_tensors_to_optimizer(ext, indices)
        self._update_interpolating_weights_for_new(new_xyz.detach())

    def recycle_components(self, dead_mask=None):
        if dead_mask is None or dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if dead_mask.sum() > int(0.05 * self._opacity.shape[0]):
            sorted_vals, indices = torch.sort(torch.abs(self.get_opacity if self.get_opacity.ndim==1 else self.get_opacity[:,0]))
            dead_indices = indices[0:int(0.05 * self._opacity.shape[0])]
        if alive_indices.shape[0] <= 0:
            return
        probs = (self.get_opacity[alive_indices] if self.get_opacity.ndim==1 else self.get_opacity[alive_indices,0])
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        (new_xyz, new_sh0, new_shN, new_opacity, new_scaling, new_rotation, new_degree, _) = self._update_params(reinit_idx, ratio=ratio)
        self._xyz[dead_indices] = new_xyz
        self._sh0[dead_indices] = new_sh0
        self._shN[dead_indices] = new_shN
        self._opacity[dead_indices] = new_opacity
        self._scaling[dead_indices] = new_scaling
        self._rotation[dead_indices] = new_rotation
        self._degree[dead_indices] = new_degree
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]
        self.replace_tensors_to_optimizer(inds=reinit_idx)
        self.replace_tensors_to_optimizer_momentum(inds=dead_indices)

    def add_components(self, cap_max):
        current_num_points = int(self._opacity.shape[0])
        target_num = min(int(cap_max), int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = (self.get_opacity if self.get_opacity.ndim==1 else self.get_opacity[:,0])
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        (new_xyz, new_sh0, new_shN, new_opacity, new_scaling, new_rotation, new_degree, new_negative) = self._update_params(add_idx, ratio=ratio)
        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling
        self.densification_postfix(new_xyz, new_sh0, new_shN, new_opacity, new_scaling, new_rotation, new_degree, new_negative, add_idx, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    def save_gaussian_sequence_to_ply(self, smpl_params_path, output_dir, frame_start=0, frame_end=None, frame_step=1, format_type='standard'):
        """
        保存Gaussian点的全局坐标序列为PLY文件

        Args:
            smpl_params_path: SMPL参数文件路径 (.npz)
            output_dir: 输出目录
            frame_start: 起始帧 (默认0)
            frame_end: 结束帧 (默认None表示到最后一帧)
            frame_step: 帧步长 (默认1)
            format_type: PLY格式类型 ('standard' 标准Gaussian Splatting格式, 'simple' 简化点云格式)
        """
        from plyfile import PlyData, PlyElement
        import os

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载SMPL参数
        smpl_data = np.load(smpl_params_path, allow_pickle=True)

        # 解析SMPL参数格式
        poses, transl, expressions, jaw_poses = self._parse_smpl_params(smpl_data)

        # 确定帧范围
        total_frames = len(poses)
        if frame_end is None:
            frame_end = total_frames
        frame_end = min(frame_end, total_frames)

        print(f"开始保存Gaussian点序列 ({format_type}格式): 帧 {frame_start} 到 {frame_end-1}, 步长 {frame_step}")
        print(f"总共 {len(range(frame_start, frame_end, frame_step))} 个PLY文件")

        for frame_idx in range(frame_start, frame_end, frame_step):
            # 设置当前帧的姿态参数
            self._set_frame_params(poses[frame_idx], transl[frame_idx],
                                 expressions[frame_idx] if expressions is not None else None,
                                 jaw_poses[frame_idx] if jaw_poses is not None else None)

            # 获取当前帧的Gaussian属性并保存
            with torch.no_grad():
                if format_type == 'standard':
                    self._save_standard_ply(output_dir, frame_idx)
                else:  # simple
                    self._save_simple_ply(output_dir, frame_idx)

            if frame_idx % 50 == 0:
                print(f"已保存帧 {frame_idx}")

        print(f"完成! 所有PLY文件已保存到: {output_dir}")

    def _parse_smpl_params(self, smpl_data):
        """解析SMPL参数文件"""
        print("Available keys in SMPL file:", list(smpl_data.keys()))

        # 处理不同的参数格式
        if 'pose' in smpl_data:
            # 格式1: 直接包含完整pose参数
            poses = smpl_data['pose']
            # 检查pose参数的维度，如果是165维（旧格式），需要扩展以包含jaw_pose和expression
            if poses.shape[1] == 165:
                # 旧格式: [global_orient(3) + body_pose(63) + jaw_pose(3) + padding(6) + hands(90)]
                # 新格式: [global_orient(3) + body_pose(63) + jaw_pose(3) + expression(10) + hands(90)]
                new_poses = []
                for i in range(len(poses)):
                    pose = poses[i]
                    # 提取各部分
                    global_orient = pose[0:3]
                    body_pose = pose[3:66]
                    jaw_pose = pose[66:69]
                    # padding = pose[69:75]  # 6维填充，丢弃
                    left_hand_pose = pose[75:120]
                    right_hand_pose = pose[120:165]
                    # 使用默认expression（10维）
                    expression = np.zeros(10, dtype=np.float32)
                    
                    # 重新组合为新格式
                    new_pose = np.concatenate([
                        global_orient, body_pose, jaw_pose, expression, 
                        left_hand_pose, right_hand_pose
                    ], axis=0)
                    new_poses.append(new_pose)
                poses = np.array(new_poses)
        elif 'global_orient' in smpl_data and 'body_pose' in smpl_data:
            # 格式2: 分离的参数，需要组合
            global_orient = smpl_data['global_orient']
            body_pose = smpl_data['body_pose']
            left_hand_pose = smpl_data.get('left_hand_pose', np.zeros((len(global_orient), 45)))
            right_hand_pose = smpl_data.get('right_hand_pose', np.zeros((len(global_orient), 45)))
            jaw_pose = smpl_data.get('jaw_pose', np.zeros((len(global_orient), 3)))
            expression = smpl_data.get('expression', np.zeros((len(global_orient), 10)))

            # 组合为完整pose参数 [global_orient(3) + body_pose(63) + jaw_pose(3) + expression(10) + hands(90)] = 169维
            poses = []
            for i in range(len(global_orient)):
                # 确保expression参数只使用前10维
                expr = expression[i]
                if len(expr) > 10:
                    expr = expr[:10]
                elif len(expr) < 10:
                    expr = np.pad(expr, (0, 10 - len(expr)))
                
                pose = np.concatenate([
                    global_orient[i],           # 3维全局旋转
                    body_pose[i],              # 63维身体姿态
                    jaw_pose[i],               # 3维下颌姿态
                    expr,                      # 10维表情参数
                    left_hand_pose[i],         # 45维左手姿态
                    right_hand_pose[i],        # 45维右手姿态
                ], axis=0)
                poses.append(pose)
            poses = np.array(poses)
        else:
            raise ValueError("SMPL参数文件格式不支持，需要包含'pose'或'global_orient'+'body_pose'")

        # 平移参数
        transl = smpl_data.get('transl', smpl_data.get('Th', np.zeros((len(poses), 3))))

        # 表情参数
        expressions = smpl_data.get('expression', None)
        if expressions is not None:
            expressions = expressions[:, :10]  # 使用前10维

        # 下颌参数
        jaw_poses = smpl_data.get('jaw_pose', None)

        return poses, transl, expressions, jaw_poses

    def _set_frame_params(self, pose, transl, expression=None, jaw_pose=None):
        """设置当前帧的参数"""
        self.smpl_poses = torch.from_numpy(pose).float()
        self.Th = torch.from_numpy(transl).float()
        self.Rh = torch.eye(3, dtype=torch.float32)  # 使用单位矩阵

        if expression is not None:
            # 确保expression是10维的
            expr_tensor = torch.from_numpy(expression).float()
            if expr_tensor.shape[0] > 10:
                expr_tensor = expr_tensor[:10]
            elif expr_tensor.shape[0] < 10:
                # 如果不足10维，用0填充
                padding = torch.zeros(10 - expr_tensor.shape[0], dtype=torch.float32)
                expr_tensor = torch.cat([expr_tensor, padding])
            self.expression = expr_tensor
        if jaw_pose is not None:
            self.jaw_pose = torch.from_numpy(jaw_pose).float()

    def _save_standard_ply(self, output_dir, frame_idx):
        """保存标准Gaussian Splatting格式的PLY文件"""
        from plyfile import PlyData, PlyElement

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # 获取球谐函数特征
        sh = self.get_sh.detach().cpu().numpy()
        f_dc = sh[:, :1, :].transpose(0, 2, 1).reshape(-1, 3)  # [N, 3] DC分量
        if sh.shape[1] > 1:
            f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(-1, (sh.shape[1]-1)*3)  # [N, rest*3]
        else:
            f_rest = np.zeros((xyz.shape[0], 0))

        # 获取并处理Gaussian属性
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self.get_cano_scaling.detach().cpu().numpy()
        rotation = self.get_cano_rotation.detach().cpu().numpy()

        # 修复Gaussian Splatting属性
        opacities, scale, rotation = self._fix_gaussian_attributes(opacities, scale, rotation)

        # 构建属性列表
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attributes.extend(['f_dc_0', 'f_dc_1', 'f_dc_2'])
        for i in range(f_rest.shape[1]):
            attributes.append(f'f_rest_{i}')
        attributes.append('opacity')
        attributes.extend(['scale_0', 'scale_1', 'scale_2'])
        attributes.extend(['rot_0', 'rot_1', 'rot_2', 'rot_3'])

        dtype_full = [(attribute, 'f4') for attribute in attributes]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # 组合所有属性
        if f_rest.shape[1] > 0:
            attributes_array = np.concatenate((xyz, normals, f_dc, f_rest,
                                             opacities.reshape(-1, 1), scale, rotation), axis=1)
        else:
            attributes_array = np.concatenate((xyz, normals, f_dc,
                                             opacities.reshape(-1, 1), scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes_array))
        el = PlyElement.describe(elements, 'vertex')

        # 保存PLY文件
        ply_path = os.path.join(output_dir, f"gaussian_frame_{frame_idx:06d}.ply")
        PlyData([el]).write(ply_path)

    def _fix_gaussian_attributes(self, opacities, scale, rotation):
        """修复Gaussian Splatting属性以确保正确显示"""

        # 1. 修复透明度：确保在[0,1]范围内，并应用sigmoid激活
        opacities = np.clip(opacities, -10, 10)  # 避免极值
        opacities = 1.0 / (1.0 + np.exp(-opacities))  # sigmoid激活

        # 2. 修复缩放：限制最大缩放值，避免过大的球
        scale = np.exp(scale)  # 通常scale是log空间的
        scale = np.clip(scale, 1e-6, 0.1)  # 限制缩放范围，0.1是一个合理的最大值

        # 3. 修复旋转：确保四元数归一化
        rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
        rotation_norm = np.maximum(rotation_norm, 1e-8)  # 避免除零
        rotation = rotation / rotation_norm

        return opacities, scale, rotation

    def _save_simple_ply(self, output_dir, frame_idx):
        """保存简化点云格式的PLY文件"""
        from utils.general_utils import storePly

        xyz_global = self.get_xyz.cpu().numpy()

        # 基于球谐函数生成颜色
        sh = self.get_sh.cpu().numpy()
        if sh.shape[1] > 0:
            sh_colors = sh[:, 0, :]  # 取第0阶球谐系数
            colors = np.clip((sh_colors + 0.5) * 255, 0, 255).astype(np.uint8)
        else:
            # 如果没有球谐函数，使用位置着色
            colors = np.clip((xyz_global - xyz_global.min()) / (xyz_global.max() - xyz_global.min()) * 255, 0, 255).astype(np.uint8)

        # 保存PLY文件
        ply_path = os.path.join(output_dir, f"gaussian_frame_{frame_idx:06d}.ply")
        storePly(ply_path, xyz_global, colors)

    def save_current_frame_to_ply(self, output_path, format_type='simple', color_mode='sh'):
        """
        保存当前帧的Gaussian点为PLY文件

        Args:
            output_path: 输出PLY文件路径
            format_type: PLY格式类型 ('standard' 标准Gaussian Splatting格式, 'simple' 简化点云格式)
            color_mode: 着色模式，仅在simple格式下有效 ('position', 'opacity', 'sh', 'uniform')
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with torch.no_grad():
            if format_type == 'standard':
                self._save_standard_ply_to_path(output_path)
            else:  # simple
                self._save_simple_ply_to_path(output_path, color_mode)

    def _save_standard_ply_to_path(self, output_path):#？？
        """保存标准Gaussian Splatting格式的PLY文件到指定路径"""
        from plyfile import PlyData, PlyElement

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # 获取球谐函数特征
        sh = self.get_sh.detach().cpu().numpy()
        f_dc = sh[:, :1, :].transpose(0, 2, 1).reshape(-1, 3)  # [N, 3] DC分量
        if sh.shape[1] > 1:
            f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(-1, (sh.shape[1]-1)*3)  # [N, rest*3]
        else:
            f_rest = np.zeros((xyz.shape[0], 0))

        # 获取并处理Gaussian属性
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self.get_cano_scaling.detach().cpu().numpy()
        rotation = self.get_cano_rotation.detach().cpu().numpy()

        # 修复Gaussian Splatting属性
        opacities, scale, rotation = self._fix_gaussian_attributes(opacities, scale, rotation)

        # 构建属性列表
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attributes.extend(['f_dc_0', 'f_dc_1', 'f_dc_2'])
        for i in range(f_rest.shape[1]):
            attributes.append(f'f_rest_{i}')
        attributes.append('opacity')
        attributes.extend(['scale_0', 'scale_1', 'scale_2'])
        attributes.extend(['rot_0', 'rot_1', 'rot_2', 'rot_3'])

        dtype_full = [(attribute, 'f4') for attribute in attributes]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # 组合所有属性
        if f_rest.shape[1] > 0:
            attributes_array = np.concatenate((xyz, normals, f_dc, f_rest,
                                             opacities.reshape(-1, 1), scale, rotation), axis=1)
        else:
            attributes_array = np.concatenate((xyz, normals, f_dc,
                                             opacities.reshape(-1, 1), scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes_array))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)

    def _save_simple_ply_to_path(self, output_path, color_mode='sh'):#？？
        """保存简化点云格式的PLY文件到指定路径"""
        from utils.general_utils import storePly

        xyz_global = self.get_xyz.cpu().numpy()

        # 根据着色模式生成颜色
        if color_mode == 'position':
            # 基于位置的着色
            colors = np.clip((xyz_global - xyz_global.min()) / (xyz_global.max() - xyz_global.min()) * 255, 0, 255).astype(np.uint8)
        elif color_mode == 'opacity':
            # 基于透明度的着色
            opacity = self.get_opacity.cpu().numpy()
            opacity_normalized = (opacity - opacity.min()) / (opacity.max() - opacity.min())
            colors = np.stack([opacity_normalized * 255, opacity_normalized * 255, opacity_normalized * 255], axis=1).astype(np.uint8)
        elif color_mode == 'sh':
            # 基于球谐函数的着色
            sh = self.get_sh.cpu().numpy()
            if sh.shape[1] > 0:
                sh_colors = sh[:, 0, :]  # 取第0阶球谐系数
                colors = np.clip((sh_colors + 0.5) * 255, 0, 255).astype(np.uint8)
            else:
                colors = np.full((len(xyz_global), 3), 128, dtype=np.uint8)
        elif color_mode == 'uniform':
            # 统一颜色 (白色)
            colors = np.full((len(xyz_global), 3), 255, dtype=np.uint8)
        else:
            # 默认使用位置着色
            colors = np.clip((xyz_global - xyz_global.min()) / (xyz_global.max() - xyz_global.min()) * 255, 0, 255).astype(np.uint8)

        storePly(output_path, xyz_global, colors)
