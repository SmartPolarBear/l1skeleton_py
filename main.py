import random

import numpy as np
import open3d as o3d

from skeleton.skeletonization import skeletonize

if __name__ == "__main__":
    points = np.load("data/default_original.npy")

    if len(points) > 5000:
        random_indices = random.sample(range(0, len(points)), 5000)
        points = points[random_indices, :]

    myCenters = skeletonize(points)

    original = o3d.geometry.PointCloud()
    original.points = o3d.utility.Vector3dVector(points)
    original.colors = o3d.utility.Vector3dVector([[0, 0.9, 0] for p in points])

    cloud = o3d.geometry.PointCloud()
    cts = [c.center for c in myCenters.myCenters]
    cloud.points = o3d.utility.Vector3dVector(cts)
    cloud.colors = o3d.utility.Vector3dVector([[0.9, 0.0, 0.0] for p in cts])

    o3d.visualization.draw_geometries([original, cloud])
