

import math
import numpy as np
import torch


def rand_point_on_mesh(vert, face, pts_num=7000, init_factor=5):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=pts_num, init_factor=init_factor)
    point_cloud = np.asarray(point_cloud.points)
    return point_cloud


def polar_decomposition_newton_schulz(A, iteration=4):
    n, m = A.shape[-2:]
    if n != m:
        raise ValueError("Matrix must be square")

    Q = A.clone()
    for _ in range(iteration):
        Q_t = Q.transpose(-2, -1)
        Q = 0.5 * (Q + torch.linalg.inv_ex(Q_t)[0])
    return Q
