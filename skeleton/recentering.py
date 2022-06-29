import numpy as np

import skeleton.utils as utils
from skeleton.center_type import CenterType

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
    normal = center.normal()
    # normal = utils.unit_vector(normal)

    if np.allclose(normal, np.zeros_like(normal)):
        return center

    if not np.isfinite(normal).all():
        return center

    p = center.center.copy()

    projected = np.array(
        [utils.project_one_point(q, p, normal) for q in neighbors if np.isfinite(q).all()])

    # visualize_result(projected, neighbors, p)

    success, cp = ellipse_center(projected)
    if not success:
        center.set_label(CenterType.REMOVED)
        return center

    nxy = normal[[0, 1]]
    diff = p[[0, 1]] - cp
    pz = -np.dot(diff, nxy) / normal[2] + p[2]

    cp = np.append(cp, pz)

    move = cp - center.center
    l_move = np.linalg.norm(move)
    if l_move > max_dist_move:
        cp = center.center + move * (max_dist_move / l_move)

    center.center = cp
    return center
