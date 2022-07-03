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

import timeit


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


class CodeTimer:
    def __init__(self, desc=None):
        self.t_start = 0
        self.t_end = 0
        self.desc = desc

    def __enter__(self):
        self.t_start = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = timeit.default_timer()
        desc = self.desc
        if desc is None:
            desc = "Time: "

        print(desc, self.t_end - self.t_start)
