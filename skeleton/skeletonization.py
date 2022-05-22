import numpy as np

import random
import sys
import time

import skeleton.center as sct
from skeleton.params import get_h0, get_density_weights
from skeleton.utils import get_local_points


def skeletonize(points, n_centers=1000, max_points=10000, max_iterations=50, try_make_skeleton=True):
    assert len(points) > n_centers

    if len(points) > max_points:
        random_indices = random.sample(range(0, len(points)), max_points)
        points = points[random_indices, :]

    h0 = get_h0(points) / 2
    h = h0

    print("h0:", h0)

    random.seed(time.time().real)

    random_centers = random.sample(range(0, len(points)), n_centers)
    centers = points[random_centers, :]

    skl_centers = sct.Centers(centers, h, maxPoints=2000)
    density_weights = get_density_weights(points, h0)

    print("Max iterations: {}, Number points: {}, Number centers: {}".format(max_iterations, len(points), len(centers)))

    last_non_branch = len(centers)
    non_change_iters = 0
    for i in range(max_iterations):

        bridge_points = 0
        non_branch_points = 0
        for center in skl_centers.myCenters:
            if center.label == 'bridge_point':
                bridge_points += 1
            if center.label == 'non_branch_point':
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
            print("Found WHOLE skeleton!")
            break

        last_non_branch = non_branch_points

        h = h + h0 / 2

    return skl_centers
