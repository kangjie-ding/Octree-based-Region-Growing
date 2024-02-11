# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/1/24 10:44

import open3d as o3d
import numpy as np
from typing import Dict, List, Set

from octree import Octree

def draw_segments(segments: List[Set[Octree]]):
    """显示算法中生成的分割区域"""
    np.random.seed(0)
    colors = [np.random.rand(3) for _ in range(len(segments))]
    clouds = []
    for i, segment in enumerate(segments):
        points = []
        pcd = o3d.geometry.PointCloud()
        for leaf in segment:
            for p in leaf.indices:
                points.append(leaf.root.cloud[p])
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(colors[i])
        clouds.append(pcd)
    o3d.visualization.draw_geometries(clouds,width=1000,height=800,left=50,top=50)


