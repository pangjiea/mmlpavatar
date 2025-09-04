import os
import argparse
import numpy as np
import torch
import smplx


def load_smplx_model(model_npz_path: str) -> smplx.SMPLX:
    """Initialize SMPL-X model from a local NPZ path.
    Mirrors init pattern used in utils/smpl_utils.py for this repo.
    """
    return smplx.SMPLX(
        model_path=model_npz_path,
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )


@torch.no_grad()
def smplx_vertices_world(model: smplx.SMPLX, pose_165: np.ndarray, beta_10: np.ndarray, Rh_3x3: np.ndarray, Th_3: np.ndarray) -> np.ndarray:
    """Compute SMPL-X vertices in world coordinates.
    - pose_165: (165,) axis-angle order used in this repo
    - beta_10: (10,)
    - Rh_3x3: (3,3) rotation matrix
    - Th_3: (3,) translation vector
    Returns: (V,3) np.float32
    """
    # Convert to torch CPU tensors
    p = torch.as_tensor(pose_165, dtype=torch.float32)
    b = torch.as_tensor(beta_10, dtype=torch.float32)

    # Split pose components as per repo convention (see test.py:get_smpl_vertices_world)
    go = p[0:3][None]                 # (1,3)
    body = p[3: 3 + 21 * 3][None]     # (1,63)
    jaw = p[3 + 21 * 3: 3 + 21 * 3 + 3][None]  # (1,3)
    # skip eyes (zeros)
    lh = p[3 + 21 * 3 + 3 + 6: 3 + 21 * 3 + 3 + 6 + 15 * 3][None]  # (1,45)
    rh = p[3 + 21 * 3 + 3 + 6 + 15 * 3: 3 + 21 * 3 + 3 + 6 + 30 * 3][None]  # (1,45)

    out = model(
        betas=b[None],
        global_orient=go,
        body_pose=body,
        jaw_pose=jaw,
        left_hand_pose=lh,
        right_hand_pose=rh,
        transl=None,
        return_verts=True,
    )
    verts = out.vertices[0].detach().cpu().float()  # (V,3)

    # Apply world transform: Rh @ v + Th (aligns with repo usage)
    Rh = torch.as_tensor(Rh_3x3, dtype=torch.float32)
    Th = torch.as_tensor(Th_3, dtype=torch.float32)
    verts_w = (Rh @ verts.T).T + Th
    return verts_w.numpy()


def write_ply_pointcloud(xyz: np.ndarray, out_path: str):
    """Write an ASCII PLY point cloud with only XYZ."""
    n = xyz.shape[0]
    header = [
        'ply',
        'format ascii 1.0',
        f'element vertex {n}',
        'property float x',
        'property float y',
        'property float z',
        'end_header'
    ]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for v in xyz:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def write_ply_mesh(xyz: np.ndarray, faces: np.ndarray, out_path: str):
    """Write an ASCII PLY triangular mesh (XYZ + faces)."""
    n_v = xyz.shape[0]
    n_f = faces.shape[0]
    header = [
        'ply',
        'format ascii 1.0',
        f'element vertex {n_v}',
        'property float x',
        'property float y',
        'property float z',
        f'element face {n_f}',
        'property list uchar int vertex_indices',
        'end_header'
    ]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for v in xyz:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def main():
    ap = argparse.ArgumentParser(description='Convert SMPL-X NPZ params to point-cloud PLY (MeshLab)')
    ap.add_argument('--npz', required=True, help='Path to ori_smpl_params.npz')
    ap.add_argument('--model', default='./smpl_model/smplx/SMPLX_NEUTRAL.npz', help='Path to SMPLX_NEUTRAL.npz')
    ap.add_argument('--out', help='Output PLY path; default next to NPZ with .ply')
    args = ap.parse_args()

    npz_path = args.npz
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    out_path = args.out or os.path.splitext(npz_path)[0] + '.ply'

    data = np.load(npz_path)
    pose = data['pose']  # (165,)
    Th = data['Th']      # (3,)
    Rh = data['Rh']      # (3,3)
    beta = data['beta']  # (10,)

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SMPL-X model not found: {model_path}")

    model = load_smplx_model(model_path)
    xyz = smplx_vertices_world(model, pose, beta, Rh, Th)
    # Prefer mesh PLY with faces if available
    faces = None
    if hasattr(model, 'faces') and isinstance(model.faces, (np.ndarray, list)):
        faces = np.asarray(model.faces, dtype=np.int32)
    elif hasattr(model, 'faces_tensor'):
        try:
            faces = model.faces_tensor.detach().cpu().numpy().astype(np.int32)
        except Exception:
            faces = None

    if faces is not None and faces.ndim == 2 and faces.shape[1] == 3:
        write_ply_mesh(xyz, faces, out_path)
        print(f"[OK] Wrote mesh PLY: {out_path}  (V={xyz.shape[0]}, F={faces.shape[0]})")
    else:
        write_ply_pointcloud(xyz, out_path)
        print(f"[OK] Wrote point cloud: {out_path}  (N={xyz.shape[0]})  [faces not available]")


if __name__ == '__main__':
    main()
