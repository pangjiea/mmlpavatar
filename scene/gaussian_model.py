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
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

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
from utils.general_utils import storePly

class GaussianModel:

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = F.normalize

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = torch.logit

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
        self.num_face_basis = self.num_basis
        self.body_pose_dim = 63

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
        self._expression = torch.zeros(10, dtype=torch.float32)
        self._jaw_pose = torch.zeros(3, dtype=torch.float32)
        self._leye_pose = torch.zeros(3, dtype=torch.float32)
        self._reye_pose = torch.zeros(3, dtype=torch.float32)
        self.face_mlp = None
        self.face_mlp_layers = None
        self.face_mlp_input_dim = None
        self.face_basis_bias = None
        self.face_gs_indices = torch.empty(0, dtype=torch.long)
        self.face_ft_indices = torch.empty(0, dtype=torch.long)
        self.face_gaussian_mask = torch.empty(0, dtype=torch.bool)
        self.face_anchor_mask = torch.empty(0, dtype=torch.bool)

        # cache
        self.cache_dict = {}

        # optimizer
        self.optimizers = None
        self.schedulers = None

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
            'face_mlp_layers': self.face_mlp_layers,
            'face_mlp_state': None,
            'face_basis_bias': self.face_basis_bias,
            'face_gs_indices': self.face_gs_indices,
            'face_gaussian_mask': self.face_gaussian_mask,
            'face_ft_indices': self.face_ft_indices,
            'face_anchor_mask': self.face_anchor_mask,

            'dxyz_bs': self.dxyz_bs,
            'sh0_bs': self.sh0_bs,
            'shN_bs': self.shN_bs,
            'scaling_bs': self.scaling_bs,
            'rotation_bs': self.rotation_bs,
            'opacity_bs': self.opacity_bs,

            'is_dxyz_bs': self.is_dxyz_bs,
            'is_gsparam_bs': self.is_gsparam_bs,
        }
        if self.face_mlp is not None:
            data['face_mlp_state'] = {k: v.detach().cpu() for k, v in self.face_mlp.state_dict().items()}
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
        self.face_mlp_layers = loader('face_mlp_layers')
        face_mlp_state = loader('face_mlp_state')
        if self.face_mlp_layers is not None and face_mlp_state is not None:
            self.face_mlp = MLP(layers_size_list=self.face_mlp_layers).cuda()
            face_mlp_state = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in face_mlp_state.items()}
            self.face_mlp.load_state_dict(face_mlp_state)
        else:
            self.face_mlp = None
        self.face_mlp_input_dim = self.face_mlp_layers[0] if self.face_mlp_layers is not None else None
        self.face_basis_bias = loader('face_basis_bias')
        self.face_gs_indices = loader('face_gs_indices')
        self.face_gaussian_mask = loader('face_gaussian_mask')
        self.face_ft_indices = loader('face_ft_indices')
        self.face_anchor_mask = loader('face_anchor_mask')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.face_basis_bias is not None:
            self.face_basis_bias = nn.Parameter(self.face_basis_bias.to(device).requires_grad_(True))
        if self.face_gs_indices is not None:
            self.face_gs_indices = self.face_gs_indices.to(device)
        else:
            self.face_gs_indices = torch.empty(0, dtype=torch.long, device=device)
        if self.face_gaussian_mask is not None:
            self.face_gaussian_mask = self.face_gaussian_mask.to(device)
        else:
            self.face_gaussian_mask = torch.empty(0, dtype=torch.bool, device=device)
        if self.face_ft_indices is not None:
            self.face_ft_indices = self.face_ft_indices.to(device)
        else:
            self.face_ft_indices = torch.empty(0, dtype=torch.long, device=device)
        if self.face_anchor_mask is not None:
            self.face_anchor_mask = self.face_anchor_mask.to(device)
        else:
            self.face_anchor_mask = torch.empty(0, dtype=torch.bool, device=device)

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
    def get_body_pose_features(self):

        if self.is_test:
            sigma_pca = 2.0
            features = self.smpl_poses_cuda[1 * 3:22 * 3][None]
            lowdim_pose_conds = self.pca.transform(features)
            std = self.pca_std
            lowdim_pose_conds = torch.maximum(lowdim_pose_conds, -sigma_pca * std)
            lowdim_pose_conds = torch.minimum(lowdim_pose_conds, sigma_pca * std)
            body_features = self.pca.inverse_transform(lowdim_pose_conds).reshape(-1)
        else:
            body_features = self.smpl_poses_cuda[3:3 * 22]  # 63维 body poses

        return body_features

    @property
    def get_face_condition_features(self):
        device = self._xyz.device if self._xyz.numel() > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        expression_cuda = torch.as_tensor(self.expression, dtype=torch.float32, device=device)
        jaw_pose_cuda = torch.as_tensor(self.jaw_pose, dtype=torch.float32, device=device)
        leye_pose_cuda = torch.as_tensor(self.leye_pose, dtype=torch.float32, device=device)
        reye_pose_cuda = torch.as_tensor(self.reye_pose, dtype=torch.float32, device=device)
        face_features = torch.cat([expression_cuda, jaw_pose_cuda, leye_pose_cuda, reye_pose_cuda], dim=0)
        return face_features

    @property
    def get_joint_features(self):
        body_features = self.get_body_pose_features
        face_features = self.get_face_condition_features
        features = torch.cat([body_features, face_features])
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
        features = self.get_body_pose_features
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

        if self.face_mlp is not None and self.face_gs_indices.numel() > 0:
            face_weights = self.get_face_basis_weights
            features = features.clone()
            features[self.face_gs_indices] = face_weights

        self.cache_dict['get_encoded_feature_gsparam_weight'] = features
        return features

    @property
    def get_face_basis_weights(self):
        if self.face_mlp is None or self.face_gs_indices.numel() == 0:
            device = self._xyz.device if self._xyz.numel() > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.empty((0, self.num_basis), device=device)

        if 'get_face_basis_weights' in self.cache_dict:
            return self.cache_dict['get_face_basis_weights']

        face_features = self.get_face_condition_features
        face_input = face_features.unsqueeze(0)
        weights = self.face_mlp(face_input).squeeze(0)
        if self.face_basis_bias is not None:
            weights = weights + self.face_basis_bias
        weights = weights.to(self._xyz.dtype)
        weights = weights.expand(self.face_gs_indices.shape[0], -1)
        self.cache_dict['get_face_basis_weights'] = weights
        return weights

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

    def _initialize_face_region(self, face_vertices, xyz, xyz_ft, data_dir):
        device = xyz.device
        print(f"[debug] initialize_face_region: face_vertices is None? {face_vertices is None}")

        if face_vertices is None or len(face_vertices) == 0:
            self.face_gs_indices = torch.empty(0, dtype=torch.long, device=device)
            self.face_gaussian_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=device)
            self.face_ft_indices = torch.empty(0, dtype=torch.long, device=device)
            self.face_anchor_mask = torch.zeros(xyz_ft.shape[0], dtype=torch.bool, device=device) if xyz_ft is not None else torch.zeros(0, dtype=torch.bool, device=device)
            return

        face_vertices_tensor = torch.as_tensor(face_vertices, dtype=torch.float32, device=device)
        print(f"[debug] face_vertices tensor shape {face_vertices_tensor.shape}")
        if face_vertices_tensor.numel() == 0:
            self.face_gs_indices = torch.empty(0, dtype=torch.long, device=device)
            self.face_gaussian_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=device)
            self.face_ft_indices = torch.empty(0, dtype=torch.long, device=device)
            self.face_anchor_mask = torch.zeros(xyz_ft.shape[0], dtype=torch.bool, device=device) if xyz_ft is not None else torch.zeros(0, dtype=torch.bool, device=device)
            return

        print(f"[debug] Running KNN for face region: xyz {xyz.shape}, xyz_ft {None if xyz_ft is None else xyz_ft.shape}")
        _, idxs_gs, _ = knn_points(
            p1=face_vertices_tensor[None],
            p2=xyz[None],
            K=1,
        )
        face_gs_idx = torch.unique(idxs_gs[0, :, 0])
        face_gaussian_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=device)
        face_gaussian_mask[face_gs_idx] = True
        self.face_gs_indices = face_gs_idx
        self.face_gaussian_mask = face_gaussian_mask

        if xyz_ft is not None and xyz_ft.shape[0] > 0:
            _, idxs_ft, _ = knn_points(
                p1=face_vertices_tensor[None],
                p2=xyz_ft[None],
                K=1,
            )
            face_ft_idx = torch.unique(idxs_ft[0, :, 0])
            face_anchor_mask = torch.zeros(xyz_ft.shape[0], dtype=torch.bool, device=device)
            face_anchor_mask[face_ft_idx] = True
            self.face_ft_indices = face_ft_idx
            self.face_anchor_mask = face_anchor_mask
            print(f"Face anchor MLP count: {int(face_anchor_mask.sum().item())} / {xyz_ft.shape[0]}")
            self._export_face_anchor_debug(xyz, xyz_ft, face_gaussian_mask, face_anchor_mask, data_dir)
        else:
            self.face_ft_indices = torch.empty(0, dtype=torch.long, device=device)
            self.face_anchor_mask = torch.zeros(0, dtype=torch.bool, device=device)

    def _export_face_anchor_debug(self, xyz, xyz_ft, face_mask, anchor_mask, data_dir):
        if data_dir is None:
            print('[debug] face debug export skipped: data_dir None')
            return
        try:
            out_dir = os.path.join(data_dir, 'gaussian')
            os.makedirs(out_dir, exist_ok=True)

            xyz_cpu = xyz.detach().cpu().numpy()
            print(f"[debug] exporting face debug ply to {out_dir}")
            colors_gs = np.zeros((xyz_cpu.shape[0], 3), dtype=np.uint8)
            colors_gs[:, 2] = 255
            face_mask_np = face_mask.detach().cpu().numpy().astype(bool)
            colors_gs[face_mask_np] = np.array([255, 0, 0], dtype=np.uint8)
            storePly(os.path.join(out_dir, 'init_gaussians_face_debug.ply'), xyz_cpu, colors_gs)

            if xyz_ft is not None and xyz_ft.shape[0] > 0:
                xyz_ft_cpu = xyz_ft.detach().cpu().numpy()
                colors_ft = np.zeros((xyz_ft_cpu.shape[0], 3), dtype=np.uint8)
                colors_ft[:, 2] = 255
                anchor_mask_np = anchor_mask.detach().cpu().numpy().astype(bool)
                colors_ft[anchor_mask_np] = np.array([255, 0, 0], dtype=np.uint8)
                storePly(os.path.join(out_dir, 'init_feature_anchors_face_debug.ply'), xyz_ft_cpu, colors_ft)
        except Exception as e:
            print(f"[warn] face anchor visualization failed: {e}")

    def create_from_pcd(self, xyz=None, t_joints=None, joint_parents=None, all_poses=None, lbs_weights_grid_info=None, xyz_vt=None, xyz_ft=None, face_vertices=None, data_dir=None):
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

        self.t_joints = torch.as_tensor(t_joints).detach().float().cpu()
        self.joint_parents = torch.as_tensor(joint_parents).detach().cpu()

        for key in all_poses: all_poses[key] = torch.as_tensor(all_poses[key]).float().cpu()
        self.all_poses = all_poses

        ginfo = lbs_weights_grid_info
        for key in ['grid', 'bbox_min', 'bbox_max', 'grid_dims']: ginfo[key] = torch.as_tensor(ginfo[key]).detach().cuda()
        self.weights_grid_info = ginfo

        # Pose encoder - 仅使用身体姿态作为输入
        models = [MLP(layers_size_list=[self.body_pose_dim, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]) for i in range(len(xyz_ft))]
        params, _ = stack_module_state(models)
        self.encoder_feat_model_meta = MLP(layers_size_list=[self.body_pose_dim, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]).to('meta')
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

        self._initialize_face_region(face_vertices, self._xyz, xyz_ft, data_dir)

        face_input_dim = int(self.expression.numel() + self.jaw_pose.numel() + self.leye_pose.numel() + self.reye_pose.numel())
        self.face_mlp_input_dim = face_input_dim
        self.face_mlp_layers = [face_input_dim, 256, 128, self.num_face_basis]
        self.face_mlp = MLP(layers_size_list=self.face_mlp_layers).cuda()
        self.face_basis_bias = nn.Parameter(torch.zeros(self.num_face_basis, dtype=torch.float32, device=self._xyz.device))

        self.prepare_interpolating_weights(xyz_ft, xyz_vt)

        self.init()

    def training_setup(self, args: Config, scene_scale):
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
        }
        if self.face_mlp is not None:
            optimizers['face_mlp'] = AdamW(self.face_mlp.parameters(), args.encoder_lr, betas, eps, decay)
        if self.face_basis_bias is not None:
            optimizers['face_basis_bias'] = AdamW([self.face_basis_bias], args.encoder_lr, betas, eps, decay)

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
        if 'face_mlp' in optimizers:
            schedulers.append(ExponentialLR(optimizers['face_mlp'], gamma=0.1 ** (1.0 / args.iterations)))
        if 'face_basis_bias' in optimizers:
            schedulers.append(ExponentialLR(optimizers['face_basis_bias'], gamma=0.1 ** (1.0 / args.iterations)))

        self.optimizers = optimizers
        self.schedulers = schedulers

    def optimizer_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        
        self.cache_dict = {}

    def render(self, cam, override_color=None, scaling_modifier=1.0, background=None):
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
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value):
        self.cache_dict = {}
        self._expression = torch.as_tensor(value, dtype=torch.float32).cpu()

    @property
    def jaw_pose(self):
        return self._jaw_pose

    @jaw_pose.setter
    def jaw_pose(self, value):
        self.cache_dict = {}
        self._jaw_pose = torch.as_tensor(value, dtype=torch.float32).cpu()

    @property
    def leye_pose(self):
        return self._leye_pose

    @leye_pose.setter
    def leye_pose(self, value):
        self.cache_dict = {}
        self._leye_pose = torch.as_tensor(value, dtype=torch.float32).cpu()

    @property
    def reye_pose(self):
        return self._reye_pose

    @reye_pose.setter
    def reye_pose(self, value):
        self.cache_dict = {}
        self._reye_pose = torch.as_tensor(value, dtype=torch.float32).cpu()

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
        # 使用与渲染一致的属性编码导出：
        # - 位置: 全局坐标 self.get_xyz
        # - 颜色: SH 系数 (DC + REST)
        # - 不透明度: 逆激活 logit(alpha)
        # - 尺度: 逆激活 log(scale)
        # - 旋转: 与LBS/全局旋转合成后的四元数，WXYZ 顺序

        # 位置
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # SH 系数
        sh = self.get_sh.detach().cpu().numpy()
        f_dc = sh[:, :1, :].transpose(0, 2, 1).reshape(-1, 3)
        if sh.shape[1] > 1:
            f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(-1, (sh.shape[1]-1)*3)
        else:
            f_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

        # 不透明度（logit）与尺度（log）
        opacities_logit = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy().reshape(-1, 1)
        scales_log = self.scaling_inverse_activation(self.get_cano_scaling).detach().cpu().numpy()

        # 合成旋转：Gweights (LBS) 与 canonical rotation，再乘全局 Rh（若有）
        rots = self.get_Gweights[:, :3, :3].contiguous()
        if self.Rh is not None:
            rots = self.Rh @ rots
        # PyTorch3D 的 quaternion_to_matrix / matrix_to_quaternion 使用 WXYZ (real-first)
        cano_R = quaternion_to_matrix(self.get_cano_rotation)
        R_total = torch.einsum('nij,njk->nik', rots, cano_R)
        rot_wxyz = matrix_to_quaternion(R_total).detach().cpu().numpy()

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
                                               opacities_logit, scales_log, rot_wxyz), axis=1)
        else:
            attributes_array = np.concatenate((xyz, normals, f_dc,
                                               opacities_logit, scales_log, rot_wxyz), axis=1)

        elements[:] = list(map(tuple, attributes_array))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)

    def export_gaussians_to_ply(self, output_path: str,
                                cam_pos: torch.Tensor | None = None,
                                format_type: str = 'standard',
                                color_mode: str = 'sh',
                                scale_mode: str = 'log',
                                scale_factor: float = 1.0,
                                scale_min: float | None = None,
                                scale_max: float | None = None,
                                opacity_mode: str = 'logit') -> None:
        """将当前帧导出为PLY。

        Args:
            output_path: 输出路径
            cam_pos: 若非空，则以视角相关颜色导出简单PLY
            format_type: 'standard' | 'simple' | 'compat'
            color_mode: simple模式颜色: 'sh'|'position'|'opacity'|'uniform'
            scale_mode: 'log'|'linear' (standard/compat 有效)
            scale_factor: 线性缩放乘子（在取log前应用）
            scale_min/scale_max: 线性缩放裁剪（在取log前应用）
            opacity_mode: 'logit'|'alpha' (standard/compat 有效)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 如果给了cam_pos，按视角颜色导出 simple PLY
        if cam_pos is not None:
            from utils.general_utils import storePly
            colors = self.get_color(cam_pos).detach().cpu().numpy()
            # get_color 返回 [N,3] 线性空间 [0,1]
            colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            xyz = self.get_xyz.detach().cpu().numpy()
            storePly(output_path, xyz, colors_u8)
            return

        if format_type == 'simple':
            return self._save_simple_ply_to_path(output_path, color_mode)

        # 标准 / 兼容 格式
        # 位置
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # SH
        sh = self.get_sh.detach().cpu().numpy()
        f_dc = sh[:, :1, :].transpose(0, 2, 1).reshape(-1, 3)
        if sh.shape[1] > 1:
            f_rest_full = sh[:, 1:, :].transpose(0, 2, 1).reshape(-1, (sh.shape[1]-1)*3)
        else:
            f_rest_full = np.zeros((xyz.shape[0], 0), dtype=np.float32)
        # pad/trim to 45 for degree 3 兼容
        if format_type in ('standard', 'compat'):
            target_rest = 45
            rest = f_rest_full
            if rest.shape[1] < target_rest:
                pad = np.zeros((rest.shape[0], target_rest - rest.shape[1]), dtype=rest.dtype)
                f_rest = np.concatenate([rest, pad], axis=1)
            else:
                f_rest = rest[:, :target_rest]
        else:
            f_rest = f_rest_full

        # 不透明度
        if opacity_mode == 'logit':
            opacity = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy().reshape(-1, 1)
        else:  # 'alpha'
            opacity = self.get_opacity.detach().cpu().numpy().reshape(-1, 1)

        # 尺度（线性→可选裁剪→乘因子→编码）
        scales_lin = self.get_cano_scaling.detach().cpu().numpy()
        if scale_min is not None or scale_max is not None:
            lo = -np.inf if scale_min is None else scale_min
            hi = np.inf if scale_max is None else scale_max
            scales_lin = np.clip(scales_lin, lo, hi)
        if scale_factor != 1.0:
            scales_lin = scales_lin * float(scale_factor)
        scales_enc = np.log(scales_lin) if scale_mode == 'log' else scales_lin

        # 旋转：与LBS/全球旋转合成，输出WXYZ
        rots = self.get_Gweights[:, :3, :3].contiguous()
        if self.Rh is not None:
            rots = self.Rh @ rots
        cano_R = quaternion_to_matrix(self.get_cano_rotation)
        R_total = torch.einsum('nij,njk->nik', rots, cano_R)
        rot_wxyz = matrix_to_quaternion(R_total).detach().cpu().numpy()

        # 写PLY
        from plyfile import PlyData, PlyElement
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
        for i in range(f_rest.shape[1]):
            attributes.append(f'f_rest_{i}')
        attributes.append('opacity')
        attributes.extend(['scale_0', 'scale_1', 'scale_2'])
        attributes.extend(['rot_0', 'rot_1', 'rot_2', 'rot_3'])
        dtype_full = [(a, 'f4') for a in attributes]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if f_rest.shape[1] > 0:
            arr = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scales_enc, rot_wxyz), axis=1)
        else:
            arr = np.concatenate((xyz, normals, f_dc, opacity, scales_enc, rot_wxyz), axis=1)
        elements[:] = list(map(tuple, arr))
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
