import numpy as np

import random
import sys
import time

import skeleton.center as sct

from skeleton.center_type import CenterType

from skeleton.params import get_density_weights
from skeleton.utils import get_local_points

import open3d as o3d
from tqdm import tqdm


class SkeletonBeforeAfterVisualizer:
    def __init__(self, skl: sct.Centers, enable=True):
        self.skl = skl
        self.enable = enable

    def __enter__(self):
        if not self.enable:
            return

        self.before_pts = self.skl.get_skeleton_points(copy=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return

        self.after_cts = self.skl.get_skeleton_points(copy=True)
        self._visualize_result()

    def _visualize_result(self):
        before_pcd = o3d.geometry.PointCloud()
        before_pcd.points = o3d.utility.Vector3dVector(self.before_pts)
        before_pcd.colors = o3d.utility.Vector3dVector([[0, 0.9, 0] for p in self.before_pts])

        after_pcd = o3d.geometry.PointCloud()
        after_pcd.points = o3d.utility.Vector3dVector([p for p in self.after_cts])
        after_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0.9] for p in self.after_cts])

        o3d.visualization.draw_geometries([before_pcd, after_pcd])

