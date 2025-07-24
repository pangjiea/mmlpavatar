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

from utils.general_utils import storePly
from utils.sh_utils import C0


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

            'dxyz_bs': self.dxyz_bs,
            'sh0_bs': self.sh0_bs,
            'shN_bs': self.shN_bs,
            'scaling_bs': self.scaling_bs,
            'rotation_bs': self.rotation_bs,
            'opacity_bs': self.opacity_bs,

            'is_dxyz_bs': self.is_dxyz_bs,
            'is_gsparam_bs': self.is_gsparam_bs,
        }
        return data

    @torch.no_grad()
    def restore(self, data):
        self._xyz = data['_xyz']
        self.xyz_offset = data['xyz_offset']
        self.dxyz_vt = data['dxyz_vt']
        self._scaling = data['_scaling']
        self._rotation = data['_rotation']
        self._opacity = data['_opacity']
        self._sh0 = data['_sh0']
        self._shN = data['_shN']
        self.sh_degree = data['sh_degree']

        self.t_joints = data['t_joints']
        self.all_poses = data['all_poses']
        self.joint_parents = data['joint_parents']

        self.nbr_gs_invdist = data['nbr_gs_invdist']
        self.nbr_gs = data['nbr_gs']
        self.nbr_vt = data['nbr_vt']
        self.nbr_gsft = data['nbr_gsft']
        self.nbr_vtft = data['nbr_vtft']
        self.nbr_gsft_wght = data['nbr_gsft_wght']
        self.nbr_vtft_wght = data['nbr_vtft_wght']

        self.xyz_vt = data['xyz_vt']
        self.xyz_ft = data['xyz_ft']

        self.num_vt_basis = data['num_vt_basis']
        self.num_basis = data['num_basis']

        self.encoder_feat_params = data['encoder_feat_params']
        self.encoder_feat_model_meta = data['encoder_feat_model_meta']

        self.dxyz_bs = data['dxyz_bs']
        self.sh0_bs = data['sh0_bs']
        self.shN_bs = data['shN_bs']
        self.scaling_bs = data['scaling_bs']
        self.rotation_bs = data['rotation_bs']
        self.opacity_bs = data['opacity_bs']

        self.is_dxyz_bs = data['is_dxyz_bs']
        self.is_gsparam_bs = data['is_gsparam_bs']

        # Derived variables
        self.smpl_poses = torch.zeros(69)  # default
        self.Rh = torch.eye(3)
        self.Th = torch.zeros(3)

        self.cache_dict = {}
        return self

    # Replace the simple getter for weights with a property that lazily
    # interpolates linear blend skinning weights from the provided SMPL/SMPLX
    # skinning field.  When `_weights` is unset this property will query
    # `interpolate_skinningfield` on the current Gaussian positions to obtain
    # per‑Gaussian per‑joint weights.  Results are cached in `_weights` to
    # avoid recomputation.  This behaviour matches the official
    # implementation and is required by downstream functions such as
    # ``get_Gweights``.
    @property
    def get_weights(self):
        # If weights have not yet been computed, interpolate them
        if self._weights is None:
            # `_xyz` stores the canonical positions of Gaussians before any
            # deformation.  We call `interpolate_skinningfield` to obtain
            # per‑Gaussian weights for each joint based on these canonical
            # positions and the precomputed `weights_grid_info`.
            xyz = self._xyz
            weights = interpolate_skinningfield(self.weights_grid_info, xyz)
            # Cache the result so that subsequent accesses reuse the same
            # tensor; this also allows overriding via `set_weights` if
            # desired.
            self._weights = weights
        return self._weights

    def set_weights(self, value):
        self._weights = value

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

        self.t_joints = torch.as_tensor(t_joints).detach().float().cpu()
        self.joint_parents = torch.as_tensor(joint_parents).detach().cpu()

        for key in all_poses: all_poses[key] = torch.as_tensor(all_poses[key]).float().cpu()
        self.all_poses = all_poses

        ginfo = lbs_weights_grid_info
        for key in ['grid', 'bbox_min', 'bbox_max', 'grid_dims']: ginfo[key] = torch.as_tensor(ginfo[key]).detach().cuda()
        self.weights_grid_info = ginfo

        # Pose encoder - 扩展输入维度支持表情: body(63) + expression(10) + jaw(3) = 76
        models = [MLP(layers_size_list=[76, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]) for i in range(len(xyz_ft))]
        params, _ = stack_module_state(models)
        self.encoder_feat_model_meta = MLP(layers_size_list=[76, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]).to('meta')
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
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        
        self.cache_dict = {}

    @torch.no_grad()
    def export_gaussians_to_ply(self, filepath: str, cam_pos: torch.Tensor = None) -> None:
        """
        Export the current set of Gaussian centers and colors to a PLY file.

        This function extracts the 3D mean positions of all Gaussians as well as
        a representative color for each point and writes them to a binary PLY
        using ``storePly`` from ``utils.general_utils``.

        Args:
            filepath (str): The path to the .ply file that will be created.
            cam_pos (torch.Tensor or None): Optional camera position from which
                view‑dependent colours should be computed. If ``None``, the method
                uses only the degree‑zero spherical harmonic coefficient (DC colour)
                to derive view‑independent colours.
        """
        # Reset the cache so that updated poses or transforms propagate to the xyz
        self.cache_dict = {}
        # Fetch Gaussian means on the CPU
        xyz_tensor = self.get_xyz
        xyz_np = xyz_tensor.detach().cpu().numpy()

        # Compute RGB colours
        if cam_pos is None:
            # Use the DC SH coefficient to compute view‑independent colour.
            sh0 = self._sh0.squeeze(1)  # shape [N,3]
            colors = sh0 * C0 + 0.5
            colors = torch.clamp(colors, 0.0, 1.0)
            colors_np = colors.detach().cpu().numpy()
        else:
            # Accept numpy or torch input for cam_pos and lift to the model device.
            if not isinstance(cam_pos, torch.Tensor):
                cam_pos = torch.as_tensor(cam_pos, dtype=torch.float32, device=xyz_tensor.device)
            # Use get_color to compute view‑dependent colour and clamp to [0,1]
            colours_tensor = self.get_color(cam_pos)
            colours_tensor = torch.clamp(colours_tensor, 0.0, 1.0)
            colors_np = colours_tensor.detach().cpu().numpy()
        # Convert [0,1] floats to uint8
        rgb = (colors_np * 255.0).astype(np.uint8)
        # Write out the PLY file
        storePly(filepath, xyz_np, rgb)

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

    # === Canonical gaussian property queries ===
    @property
    def get_cano_scaling(self):
        """Return per‑Gaussian scaling factors in canonical pose.

        The learnable parameter ``_scaling`` is exponentiated via
        ``scaling_activation`` to obtain positive scale values.  When Gaussian
        parameter basis functions are enabled (``is_gsparam_bs``), the scales are
        modulated by the encoded features and the learned ``scaling_bs``.
        """
        # Check cache first
        if 'get_cano_scaling' in self.cache_dict:
            return self.cache_dict['get_cano_scaling']
        if not self.is_gsparam_bs:
            scaling = self.scaling_activation(self._scaling)
        else:
            features = self.get_encoded_feature_gsparam_weight
            # Modulate each dimension of the scale using the features and basis weights
            dscaling = torch.einsum('nc,ncl->nl', features, self.scaling_bs)
            scaling = self._scaling + dscaling
            scaling = self.scaling_activation(scaling)
        self.cache_dict['get_cano_scaling'] = scaling
        return scaling

    @property
    def get_rigid_transform(self):
        """Compute the rigid transforms for each skeletal joint.

        This method converts the current SMPL/SMPLX pose parameters
        ``smpl_poses`` into rotation matrices, applies the forward
        kinematics to obtain absolute transforms for each joint, and then
        multiplies by the inverse of the canonical transforms ``Ac_inv`` to
        produce transforms relative to the T‑pose.  The result is a list
        containing the raw rotation matrices and the rigid transforms ``G``.
        """
        if 'get_rigid_transform' in self.cache_dict:
            return self.cache_dict['get_rigid_transform']
        # Convert pose from axis‑angle to rotation matrices on CPU for the numba function
        pose = self.smpl_poses.numpy()
        joints = self.t_joints.numpy()
        parent = self.joint_parents.numpy()
        Ac_inv = self.Ac_inv.numpy()
        rots = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().astype(np.float32)
        A = rigid_transform_numba(rots, joints, parent)
        G = np.matmul(A, Ac_inv)
        data = [torch.as_tensor(d).cuda(non_blocking=True) for d in [rots, G]]
        self.cache_dict['get_rigid_transform'] = data
        return data

    @property
    def get_Gweights(self):
        """Return the weighted combination of rigid transforms for each Gaussian.

        Each Gaussian has per‑joint weights stored in ``get_weights``.  These
        weights are used to blend the per‑joint rigid transforms ``G`` into
        a single transform matrix for each Gaussian via a weighted sum.
        """
        if 'get_Gweights' in self.cache_dict:
            return self.cache_dict['get_Gweights']
        G = self.get_rigid_transform[1]
        G_weight = torch.einsum('vp,pij->vij', self.get_weights, G)
        self.cache_dict['get_Gweights'] = G_weight
        return G_weight

    @property
    def get_cano_rotation(self):
        """Return per‑Gaussian rotation quaternions in canonical pose.

        Quaternions are stored in ``_rotation``; ``rotation_activation``
        normalizes them to unit length.  When basis functions are used,
        quaternions are modulated by the encoded features and ``rotation_bs``.
        """
        if not self.is_gsparam_bs:
            rotation = self.rotation_activation(self._rotation)
        else:
            features = self.get_encoded_feature_gsparam_weight
            drotation = torch.einsum('nc,ncl->nl', features, self.rotation_bs)
            rotation = self._rotation + drotation
            rotation = self.rotation_activation(rotation)
        return rotation

    def get_covariance(self, scaling_modifier: float = 1.0):
        """Compute full covariance matrices for each Gaussian in world coordinates.

        This method converts the canonical scale and rotation parameters into
        3×3 covariance matrices via ``quat_scale_to_covar_preci``.  The
        resulting covariances are then rotated by the per‑Gaussian blend
        weights and optionally by the global rotation ``Rh``.  Setting
        ``scaling_modifier`` allows the caller to uniformly scale the
        Gaussians for, e.g., visualization.

        Args:
            scaling_modifier (float): A multiplicative factor applied to the
                scales before converting to covariance.
        Returns:
            torch.Tensor: A tensor of shape ``(N, 3, 3)`` containing the
                covariance matrices for each Gaussian.
        """
        # Extract the rotation part of the blended transforms
        rots = self.get_Gweights[:, :3, :3].contiguous()
        covs = quat_scale_to_covar_preci(
            quats=self.get_cano_rotation,
            scales=self.get_cano_scaling * scaling_modifier,
            compute_preci=False,
        )[0]
        if self.Rh is not None:
            rots = self.Rh @ rots
        covs = rots @ covs @ rots.transpose(-1, -2)
        return covs

    @property
    def get_joint_features(self):
        """Assemble joint feature vector for the pose encoder.

        The feature vector comprises the body pose (63 dimensions) plus
        facial expression (10) and jaw pose (3), resulting in a 76‑dimensional
        vector.  In test mode, a PCA model is applied to project and clamp
        the pose before reconstructing the full body pose.
        """
        if self.is_test:
            sigma_pca = 2.0
            features = self.smpl_poses_cuda[1 * 3 : 22 * 3][None]
            lowdim_pose_conds = self.pca.transform(features)
            std = self.pca_std
            lowdim_pose_conds = torch.maximum(lowdim_pose_conds, -sigma_pca * std)
            lowdim_pose_conds = torch.minimum(lowdim_pose_conds, sigma_pca * std)
            body_features = self.pca.inverse_transform(lowdim_pose_conds).reshape(-1)
        else:
            body_features = self.smpl_poses_cuda[3 : 3 * 22]
        expression_cuda = self.expression.cuda()
        jaw_pose_cuda = self.jaw_pose.cuda()
        features = torch.cat([body_features, expression_cuda, jaw_pose_cuda])
        return features
