import numpy as np

import skeleton.utils as utils

from skimage.measure import EllipseModel

import open3d as o3d


def ellipse_center(projected):
    xy = projected[:, [0, 1]]
    ell = EllipseModel()
    if not ell.estimate(xy):
        return False, None

    xc, yc, _, _, _ = ell.params
    return True, np.array([xc, yc])


def visualize_result(projected, neighbors, p):
    prj = o3d.geometry.PointCloud()
    prj.points = o3d.utility.Vector3dVector(projected)
    prj.colors = o3d.utility.Vector3dVector([[0, 0.9, 0] for p in projected])

    original = o3d.geometry.PointCloud()
    original.points = o3d.utility.Vector3dVector([p for p in neighbors])
    original.colors = o3d.utility.Vector3dVector([[0, 0, 0.9] for p in neighbors])

    cloud = o3d.geometry.PointCloud()
    cts = [p]
    cloud.points = o3d.utility.Vector3dVector(cts)
    cloud.colors = o3d.utility.Vector3dVector([[0.9, 0.0, 0.0] for _ in cts])

    o3d.visualization.draw_geometries([prj, original, cloud])


def recenter_around(center, neighbors, max_dist_move):
    normal = center.eigen_vectors[:, 0]
    # normal = utils.unit_vector(normal)

    if not normal.any():
        return center

    if np.isnan(normal).any() or np.isinf(normal).any():
        return center

    p = center.center

    if np.isnan(normal).any() or np.isinf(normal).any():
        return center

    projected = np.array(
        [utils.project_one_point(q, p, normal) for q in neighbors if np.isfinite(q).all()])

    # visualize_result(projected, neighbors, p)

    success, cp = ellipse_center(projected)
    if not success:
        # FIXME: use bounding box instead
        print("Cannot fit with ellipse!")
        return center

    nxy = normal[[0, 1]]
    diff = p[[0, 1]] - cp
    pz = -np.dot(diff, nxy) / normal[2] + p[2]

    cp = np.append(cp, pz)

    if np.linalg.norm(center.center - cp) > max_dist_move:
        return center

    center.center = cp
    return center
