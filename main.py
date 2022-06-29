import random

import numpy as np
import open3d as o3d

from skeleton.skeletonization import skeletonize

if __name__ == "__main__":
    points = np.load("data/default_original.npy")
    # points = np.load("data/simple_tree.npy")

    # pcd = o3d.io.read_point_cloud("data/2_2D_Leaf.ply", format='ply')
    # pcd = o3d.io.read_point_cloud("data/9_GLady.ply", format='ply')
    # points = np.asarray(pcd.points)

    myCenters = skeletonize(points, n_centers=1500, downsampling_rate=1)

    if len(points) > 5000:
        random_indices = random.sample(range(0, len(points)), 5000)
        points = points[random_indices, :]

    original = o3d.geometry.PointCloud()
    original.points = o3d.utility.Vector3dVector(points)
    original.colors = o3d.utility.Vector3dVector([[0, 0.9, 0] for p in points])

    cloud = o3d.geometry.PointCloud()
    cts = myCenters.get_skeleton_points()
    # cts = myCenters.get_all_centers()

    cloud.points = o3d.utility.Vector3dVector(cts)
    cloud.normals = o3d.utility.Vector3dVector([c.normal_vector() for c in myCenters.myCenters if c.label != 4])
    cloud.colors = o3d.utility.Vector3dVector([[0.9, 0.0, 0.0] for p in cts])

    # o3d.visualization.draw_geometries([original, cloud], point_show_normal=True)
    o3d.visualization.draw_geometries([cloud], point_show_normal=True)
