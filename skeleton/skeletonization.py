import numpy as np

import random
import sys
import time

import skeleton.center as sct

from skeleton.center_type import CenterType

from skeleton.params import get_density_weights
from skeleton.utils import get_local_points

from skeleton.debug import SkeletonBeforeAfterVisualizer
from typing import Final


def skeletonize(points, n_centers=1000,
                max_iterations=50,
                dh=2.0,
                sigma_smoothing_k=5,
                error_tolerance=1e-5,
                downsampling_rate=0.5,
                try_make_skeleton=True,
                recenter_knn=200,
                max_points=None):
    assert len(points) > n_centers
    assert len(points) > recenter_knn

    if max_points is not None and len(points) > max_points:
        print("Down sampling the original point cloud")
        random_indices = random.sample(range(0, len(points)), max_points)
        points = points[random_indices, :]

    # random.seed(int(time.time()))
    random.seed(3074)

    skl_centers = sct.Centers(points=points, center_count=n_centers, smoothing_k=sigma_smoothing_k)

    # for i in range(len(skl_centers.myCenters)):
    #     skl_centers.myCenters[i].set_label(CenterType.BRANCH)
    # return skl_centers

    h = h0 = skl_centers.get_h0()
    print("h0:", h0)

    hd: Final[float] = h0 / 2
    density_weights = get_density_weights(points, hd)

    print("Max iterations: {}, Number points: {}, Number centers: {}".format(max_iterations, len(points),
                                                                             len(skl_centers.centers)))

    last_non_branch = len(skl_centers.centers)
    non_change_iters = 0
    for i in range(max_iterations):
        bridge_points = len([1 for c in skl_centers.myCenters if c.label == CenterType.BRIDGE])
        non_branch_points = len([1 for c in skl_centers.myCenters if c.label == CenterType.NON_BRANCH])

        print("\n\nIteration:{}, h:{}, bridge_points:{}\n\n".format(i, round(h, 3), bridge_points))

        last_error = 0
        for j in range(50):  # magic number. do contracting at most 30 times
            error = skl_centers.contract(h, density_weights)
            skl_centers.update_properties()

            if np.abs(error - last_error) < error_tolerance:
                break

            last_error = error

        if try_make_skeleton:
            skl_centers.find_connections()

        print("Non-branch:", non_branch_points)

        if non_branch_points == last_non_branch:
            non_change_iters += 1
        elif non_branch_points < last_non_branch:
            non_change_iters = 0

        if non_change_iters >= 5:
            print("Cannot make more branch points")
            break

        if non_branch_points == 0:
            print("Found whole skeleton!")
            break

        last_non_branch = non_branch_points

        h = h + h0 / dh

    with SkeletonBeforeAfterVisualizer(skl_centers, enable=True):
        if recenter_knn > 0:
            skl_centers.recenter(downsampling_rate=downsampling_rate, knn=recenter_knn)

    return skl_centers
