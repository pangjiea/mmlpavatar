
from dataclasses import dataclass

@dataclass
class ConfigTest:
    cam_ids = [0]
    num_frame = 500
    begin_ith_frame = 0
    frame_interval = 1
    image_scaling = 1

@dataclass
class Config:
    seed = 0
    ip: str
    port: int
    out_dir: str
    data_dir: str
    detect_anomaly = False
    test_iterations = []
    save_iterations = []
    checkpoint_iterations = []

    smpl_pkl_path = './smpl_model/smplx/SMPLX_NEUTRAL.npz'

    background = [1, 1, 1]
    random_background = True

    # dataset
    train_cam_ids = []
    num_train_frame = 100
    begin_ith_frame = 0
    frame_interval = 1
    image_scaling = 1
    data_in_memory = False

    test: ConfigTest

    # optimization
    iterations = 800_000
    optimizer = 'adam'           # 'adam' | 'sghmc'
    renderer = 'gsplat'          # 'gsplat' | 'sss'
    position_lr = 0.00016
    opacity_lr = 0.0005
    scaling_lr = 0.0005
    rotation_lr = 0.0005
    color_lr = 0.0005
    xyz_offset_lr = 0.001
    encoder_lr = 0.0005
    degree_lr = 5e-4             # SSS degree nu lr
    negative_lr = 1e-4           # SSS negative lr

    iteration_sh_degree = 250000

    # loss
    lambda_lpips = 0.1
    iteration_lpips = 6000
    iteration_lpips_random_patch = 300000
    lambda_scaling = 0.1
    scaling_threshold = 0.01
    lambda_dxyz_smooth = 0.1
    # SSS regularization (optional)
    sss_scale_reg = 0.0
    sss_opacity_reg = 0.0
    
    init_num_gs = 200_000
    # SSS densification
    cap_max = 2_000_000
    opacity_threshold = 1e-3
    densify_from_iter = 1000
    densify_until_iter = 300000
    densification_interval = 100

    # anchor points and control points
    num_verts = 10000
    num_features = 300

    # iteration to start optimize the basis
    iteration_dxyz_basis = 2000
    iteration_gsparam_basis = 2000

    # face-region loss
    enable_face_loss = True
    lambda_face_l1 = 0.0
    lambda_face_lpips = 0.0
    face_mask_method = 'smplx'  # 'heuristic' | 'smplx' | 'detector'
    face_roi_top_frac = 0.15      # fraction from top of person bbox to start ROI
    face_roi_height_frac = 0.22   # fraction of person bbox height for ROI height
    face_roi_width_frac = 0.35    # fraction of person bbox width for ROI width
    face_min_size = 32            # minimum face bbox size in pixels
    # smplx mask params
    face_smplx_radius_scale = 2.2  # multiplier on neck-head distance to set ROI radius
    face_smplx_ax_min_frac = 0.25  # fraction along neck-head axis to start selecting verts
    face_smplx_ax_max_frac = 2.0   # fraction along neck-head axis to stop selecting verts
