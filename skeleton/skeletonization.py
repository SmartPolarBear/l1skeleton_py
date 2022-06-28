import numpy as np

import random
import sys
import time

import skeleton.center as sct

from skeleton.center_type import CenterType

from skeleton.params import get_h0, get_density_weights
from skeleton.utils import get_local_points

import open3d as o3d


class SkeletonBeforeAfterVisualizer:
    def __init__(self, skl: sct.Centers, enable=True):
        self.skl = skl
        self.enable = enable

    def __enter__(self):
        if not self.enable:
            return

        self.before_pts = self.skl.get_bare_points(copy=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return

        self.after_cts = self.skl.get_bare_points(copy=True)
        self._visualize_result()

    def _visualize_result(self):
        before_pcd = o3d.geometry.PointCloud()
        before_pcd.points = o3d.utility.Vector3dVector(self.before_pts)
        before_pcd.colors = o3d.utility.Vector3dVector([[0, 0.9, 0] for p in self.before_pts])

        after_pcd = o3d.geometry.PointCloud()
        after_pcd.points = o3d.utility.Vector3dVector([p for p in self.after_cts])
        after_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0.9] for p in self.after_cts])

        o3d.visualization.draw_geometries([before_pcd, after_pcd])


def skeletonize(points, n_centers=1000,
                max_points=10000,
                max_iterations=50,
                try_make_skeleton=True,
                recenter_knn=200):
    assert len(points) > n_centers

    if len(points) > max_points:
        random_indices = random.sample(range(0, len(points)), max_points)
        points = points[random_indices, :]

    h0 = get_h0(points) / 2
    h = h0

    print("h0:", h0)

    # random.seed(int(time.time()))
    random.seed(3074)

    random_centers = random.sample(range(0, len(points)), n_centers)
    centers = points[random_centers, :]

    skl_centers = sct.Centers(centers, points, h, maxPoints=2000)
    density_weights = get_density_weights(points, h0)

    print("Max iterations: {}, Number points: {}, Number centers: {}".format(max_iterations, len(points), len(centers)))

    last_non_branch = len(centers)
    non_change_iters = 0
    for i in range(max_iterations):

        bridge_points = 0
        non_branch_points = 0
        for center in skl_centers.myCenters:
            if center.label == CenterType.BRIDGE:
                bridge_points += 1
            if center.label == CenterType.NON_BRANCH:
                non_branch_points += 1

        sys.stdout.write("\n\nIteration:{}, h:{}, bridge_points:{}\n\n".format(i, round(h, 3), bridge_points))

        centers = skl_centers.centers

        last_error = 0
        for j in range(30):
            local_indices = get_local_points(points, centers, h)
            error = skl_centers.contract(points, local_indices, h, density_weights)
            skl_centers.update_properties()

            if np.abs(error - last_error) < 0.001:
                break

            last_error = error

        if try_make_skeleton:
            skl_centers.find_connections()

        print("Non-branch:", non_branch_points)

        if non_branch_points == last_non_branch:
            non_change_iters += 1
        elif non_branch_points < last_non_branch:
            non_change_iters = 0

        if non_change_iters >= 10:
            print("Cannot make more branch points")
            break

        if non_branch_points == 0:
            print("Found whole skeleton!")
            break

        last_non_branch = non_branch_points

        h = h + h0 / 2

    with SkeletonBeforeAfterVisualizer(skl_centers):
        if recenter_knn > 0:
            skl_centers.recenter(knn=recenter_knn)

    return skl_centers
