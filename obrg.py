# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/1/23 10:43

from collections import deque
import open3d as o3d
from math import pi
import numpy as np
from tqdm import tqdm
from time import time
from typing import Dict, List, Set

from aux_formula import ang_div, dist, cal_evaluation
from octree import Octree, VIEWPOINT
from settings import Settings
from visualization import draw_segments


def obrg(O: Octree, settings) -> List[Set[Octree]]:
    R: List[Set[Octree]] = list()
    a = O.leaves
    a.sort(key=lambda x: x.residual)
    A = deque(a)
    visited = dict()
    while len(A) > 0:
        R_c: Set[Octree] = set()
        S_c: Set[Octree] = set()
        v_min = A.popleft()
        #考虑这段代码的适合场景
        if v_min.residual > settings.res_th:
            break
        S_c.add(v_min)
        R_c.add(v_min)
        while len(S_c) > 0:
            v_i = S_c.pop()
            B_c = v_i.get_neighbors()
            for v_j in B_c:
                ang = ang_div(v_i.normal, v_j.normal)
                if v_j in A and ang <= settings.ang_th:
                    R_c.add(v_j)
                    A.remove(v_j)
                    if v_j.residual < settings.res_th:
                        S_c.add(v_j)
        # 下面这段代码放置的位置需要考虑
        m = sum([len(l.indices) for l in R_c])
        if m >= settings.min_segment:
            for l in R_c:
                l.is_allocated = True
            R.append(R_c)
    return sorted(R, key=lambda x: sum([len(y.indices) for y in x]), reverse=True)

def check_planarity(R_i: Set[Octree], settings):
    """检查分割块是否足够平坦，以作为不同refinement的依据"""
    pcd = o3d.geometry.PointCloud()
    points = []
    for leaf in R_i:
        tmp = [leaf.root.cloud[i] for i in leaf.indices]
        points = points + tmp
    pcd.points = o3d.utility.Vector3dVector(points)
    mean_point, con_matrix = pcd.compute_mean_and_covariance()
    w, v = np.linalg.eig(con_matrix)
    w_indices = np.argsort(w)
    R_i_normal = v[:, w_indices[0]]
    #修正法向量方向
    if np.inner(R_i_normal, VIEWPOINT - mean_point) < 0:
        R_i_normal = -R_i_normal
    d = mean_point[0] * R_i_normal[0] + mean_point[1] * \
        R_i_normal[1] + mean_point[2] * R_i_normal[2]
    planar_sign = 0
    for point in points:
        if dist(point, R_i_normal, d) < settings.dist_th:
            planar_sign += 1
    return (planar_sign/len(points)) > 0.7, R_i_normal, d

def boundary_voxel_judgemnet(leaf, cluster: Set[Octree]) -> bool:
    """判断聚类内体素是否为边界体素"""
    assert leaf.is_leaf
    neighbors = leaf.get_neighbors()
    for neighbor in neighbors:
        if neighbor not in cluster:
            neighbors.remove(neighbor)
    return len(neighbors) < 8

def extract_boundary_voxels(cluster: Set[Octree]) -> Set[Octree]:
    """提取聚类中的边界体素"""
    boundary = set([leaf for leaf in cluster if
                    boundary_voxel_judgemnet(leaf, cluster)])
    return boundary

def fast_refinement(O: Octree, R_i: Set[Octree], V_b: Set[Octree],
                    R_i_normal, d, settings) -> Set[Octree]:
    """若粗糙聚类足够平坦，进行快速修正算法"""
    if len(V_b) == 0:
        return R_i
    S = V_b.copy()
    to_be_added: Set[int] = set()
    visited = set()
    while len(S) > 0:
        v_j = S.pop()
        visited.add(v_j)
        B = v_j.get_neighbors()
        for v_k in B:
            flag = False
            if not v_k.is_allocated:
                for index in v_k.indices:
                    if dist(v_k.root.cloud[index], R_i_normal, d) < settings.dist_th:
                        to_be_added.add(index)
                        if not flag:
                            if v_k not in visited:
                                S.add(v_k)
                                flag = True
    tmp = V_b.pop()
    for index in to_be_added:
        tmp.indices.append(index)
    V_b.add(tmp)
    return R_i.union(V_b)


def general_refinement(O: Octree, R_i: Set[Octree], V_b: Set[Octree], settings) -> None:
    """若粗糙聚类不够平坦，则进行一般修正算法"""
    if len(V_b) == 0:
        return R_i
    seed_points = []
    unallocated_points = []
    cloud = o3d.geometry.PointCloud()
    added_indices : Set[int] = set()
    points_index = dict()
    points_leaf = dict()
    for leaf in V_b:
        for index in leaf.indices:
            seed_points.append(leaf.root.cloud[index])
            points_index[tuple(leaf.root.cloud[index])] = index
        for index,leaf in leaf.get_buffer_zone_points().items():
            unallocated_points.append(leaf.root.cloud[index])
            points_index[tuple(leaf.root.cloud[index])] = index
            points_leaf[tuple(leaf.root.cloud[index])] = leaf
    cloud.points = o3d.utility.Vector3dVector(seed_points+unallocated_points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    #开始点级别的区域生长
    while len(seed_points) > 0:
        point = seed_points.pop()
        [_,idx,_] = kdtree.search_knn_vector_3d(np.array(point),20)
        for i in idx:
            index = points_index[tuple(cloud.points[i])]
            if O.cloud[index] in unallocated_points:
                ang = ang_div(O.normals[index], np.array(point))
                if ang < settings.ang_th:
                    added_indices.add(index)
                    unallocated_points.remove(O.cloud[index])
                    res = dist(O.cloud[index], points_leaf[tuple(O.cloud[index])].normal,
                               points_leaf[tuple(O.cloud[index])].d)
                    if res < settings.res_th:
                        seed_points.append(O.cloud[index])
    tmp = V_b.pop()
    for index in added_indices:
        tmp.indices.append(index)
    V_b.add(tmp)
    return R_i.union(V_b)

def refinement(O: Octree, R: List[Set[Octree]], settings) -> List[Set[Octree]]:
    """总的修正算法"""
    complete_segments: List[Set[Octree]] = []
    for R_i in R:
        is_planar,R_i_normal,d = check_planarity(R_i, settings)
        V_b = extract_boundary_voxels(R_i)
        if is_planar:
            complete_segments.append(fast_refinement(O, R_i, V_b, R_i_normal, d, settings))
        else:
            complete_segments.append(general_refinement(O, R_i, V_b, settings))
    return complete_segments

def obrg_calculation(cloud_path: str, settings, draw = False,timing = False, evaluation = False):
    """基于八叉树区域生长的整体算法"""
    points = np.loadtxt(cloud_path, dtype=float, usecols=(0, 1, 2)).tolist()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    oc = Octree(points, center=bb.get_center())
    #创建八叉树
    start_time = time()
    oc.create(settings)
    end_time = time()
    octree_creation_cost = end_time - start_time
    #基于八叉树的区域生长
    start_time = time()
    incomplete_segments = obrg(oc, settings)
    end_time = time()
    region_growing_cost = end_time - start_time
    #修正过程
    start_time = time()
    complete_segments = refinement(oc, incomplete_segments, settings)
    end_time = time()
    refinement_cost = end_time - start_time
    #显示分割图像、执行时间以及各项指标
    if draw:
        draw_segments(complete_segments)
    if timing:
        print(f"cost for creating octree:{octree_creation_cost} seconds")
        print(f"cost for octree-based region growing phase:{region_growing_cost} seconds")
        print(f"cost for refinement phase:{refinement_cost} seconds")
        print(f"the overall executing time is {octree_creation_cost+region_growing_cost+refinement_cost} seconds")
    if evaluation:
        # 计算precision,recall和F1-score
        reference_data0 = np.loadtxt(cloud_path,
                                    dtype=float, usecols=(0, 1, 2, 6)).tolist()
        reference_data = np.array(reference_data0)
        reference_data = reference_data[np.lexsort(-reference_data.T)]
        ##先根据标签，将参考数据集进行分块
        reference_segs = []
        i = 0
        while i < len(reference_data):
            seg = set()
            seg.add(tuple(reference_data[i][0:3]))
            current_flag = reference_data[i][3]
            if i < len(reference_data)-1:
                while current_flag == reference_data[i + 1][3]:
                    seg.add(tuple(reference_data[i + 1][0:3]))
                    i += 1
                    if i==len(reference_data)-1:
                        break
            i+=1
            reference_segs.append(seg)
        # #得到对应评估指标
        precision, recall, F1_score, mIoU = cal_evaluation(complete_segments, reference_segs
                                                           , reference_data0)
        print(f"the precision is :{round(precision*100,2)}%")
        print(f"the recall is :{round(recall*100,2)}%")
        print(f"the F1-score is :{round(F1_score,4)}")
        print(f"the mIoU is :{round(mIoU,4)}")
    return octree_creation_cost + region_growing_cost + refinement_cost

if __name__ == '__main__':
    """测试代码"""
    points = np.loadtxt('C:/Users/92595/Desktop/data_of_point_cloud/building-Cloud.txt', dtype=float, usecols=(0, 1, 2)).tolist()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    oc = Octree(points, center=bb.get_center())
    settings = Settings(residual=0.05,res_th=0.05,dist_th=1)
    oc.create(settings)
    # oc.draw_leaves()
    incomplete_segments = obrg(oc, settings)
    # draw_segments(incomplete_segments)
    # complete_segments = refinement(oc, incomplete_segments, settings)
    draw_segments(incomplete_segments)



