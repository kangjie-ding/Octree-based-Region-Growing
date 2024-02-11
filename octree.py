# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/1/23 10:34

from itertools import product
from typing import List
import numpy as np
import open3d as o3d
from scipy import linalg as LA

from aux_formula import dist
from octree_support import partition_judegement
from settings import Settings

centers = []
#视野点：用于修正法向量
VIEWPOINT = np.asarray([-26.644233, 18.825099, 128.247082]) #建筑物点云



class Octree:
    """八叉树的一个类"""
    def __init__(self, cloud=None, center=None) -> None:
        if cloud is not None:
            self.cloud = cloud
        self.normals = None
        self.root = self
        self.parent = None
        self.level = 0
        self.leaves: List[Octree] = []
        self.indices = []
        self.is_leaf = False
        self.is_full = True #默认节点是满的
        self.normal = np.array([0, 0, 0])
        self.residual = float('inf')
        if self.level == 0:
            self.leaf_centers = dict()
        self.d = 0
        self.num_nb = 0
        self.is_allocated = False
        if cloud is not None:
            # 考虑下面代码是否要保留
            minimum = [float('inf')] * 3
            maximum = [-float('inf')] * 3
            for i, point in enumerate(cloud):
                x, y, z = point
                minimum[0] = min(minimum[0], x)
                minimum[1] = min(minimum[1], y)
                minimum[2] = min(minimum[2], z)
                maximum[0] = max(maximum[0], x)
                maximum[1] = max(maximum[1], y)
                maximum[2] = max(maximum[2], z)
                self.indices.append(i)
            x = (minimum[0] + maximum[0]) / 2
            y = (minimum[1] + maximum[1]) / 2
            z = (minimum[2] + maximum[2]) / 2
            # 这里center是否有歧义
            if center is not None:
                self.center = np.array([*center])
            else:
                self.center = np.array([x, y, z])
            self.size = max(maximum[0] - minimum[0], max(maximum[1] -
                                                         minimum[1], maximum[2] - minimum[2]))
        self.children = [None] * 8

    def __hash__(self) -> int:
        return hash(tuple(np.around(np.array(self.center),decimals=4)))

    def __eq__(self, __o: object) -> bool:
        return id(self) == id(__o)

    def create(self, settings):
        """创建八叉树"""
        if len(self.indices) <= settings.lower_lim_num:
            self.is_full = False
            return
        # calculate saliency feature
        self._cal_feature()
        if self.level >= settings.max_level or self.residual < settings.residual:
            self.is_leaf = True
            self.root.leaves.append(self)
            self.root.leaf_centers[tuple(np.around(self.center, decimals=4))] = self
            self.children = []
            return
        newSize = self.size / 2
        difference =newSize / 2
        #按照八个卦限的顺序
        new_centers = [
            [self.center[0] + difference, self.center[1] +
                  difference, self.center[2] + difference],
            [self.center[0] - difference, self.center[1] +
                  difference, self.center[2] + difference],
            [self.center[0] - difference, self.center[1] -
                  difference, self.center[2] + difference],
            [self.center[0] + difference, self.center[1] -
                  difference, self.center[2] + difference],
            [self.center[0] + difference, self.center[1] +
                  difference, self.center[2] - difference],
            [self.center[0] - difference, self.center[1] +
                  difference, self.center[2] - difference],
            [self.center[0] - difference, self.center[1] -
                  difference, self.center[2] - difference],
            [self.center[0] + difference, self.center[1] -
                  difference, self.center[2] - difference]]
        for c in new_centers:
            centers.append(c)
        # 将new_centers列表各值依次映射到参数c中，即得到八个孩子
        self.children = list(map(lambda c: Octree._create_child(
            self, c, newSize), new_centers))
        # 根据坐标与中心比较，将点索引分到八个孩子中
        self._partition()
        for child in self.children:
            child.create(settings)

    def _partition(self):
        """根据坐标与中心比较，将点索引分到八个孩子中"""
        for i in self.indices:
            point = np.array((self.root.cloud[i]))
            index = partition_judegement(point, self)
            self.children[index].indices.append(i)


    @staticmethod
    def _create_child(parent, center, size):
        """创建孩子节点"""
        global IDS
        child = Octree()
        child.parent = parent
        #孩子和父亲的根都是一样的
        child.root = parent.root
        child.level = parent.level+1
        child.center = center
        child.size = size
        return child

    def _cal_feature(self):
        """如果是根节点则计算各点的法向量"""
        if self.level == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.cloud)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30))
            self.normals = np.asarray(pcd.normals)
        """计算节点整体法向量和残差值"""
        pcd_inliers = o3d.geometry.PointCloud()
        pcd_inliers.points= o3d.utility.Vector3dVector\
            ([self.root.cloud[i] for i in self.indices])
        mean_point, con_matrix = pcd_inliers.compute_mean_and_covariance()
        w, v = np.linalg.eig(con_matrix)
        w_indices = np.argsort(w)
        self.normal = v[:,w_indices[0]]
        #修正法向量方向
        if np.inner(self.normal,VIEWPOINT-mean_point) < 0:
            self.normal = -self.normal
        p_d = mean_point[0]*self.normal[0] + mean_point[1] * \
              self.normal[1] + mean_point[2]*self.normal[2]
        self.d = p_d
        D = 0
        for inlier in pcd_inliers.points:
            d = dist(inlier, self.normal, p_d)
            D += (d**2)
        D = D/len(self.indices)
        self.residual = D**0.5

    def search_point(self, point) -> bool:
        """通过八叉树数据结构查询该点是否在点云数据中"""
        current_node = self.root
        while(current_node.children != [None]*8):
            index = partition_judegement(point, current_node)
            current_node = current_node.children[index]
        for index in current_node.indices:
            if np.array(point).all() == np.array(current_node.root.cloud[index]).all():
                return True
        return False

    #看看之后能不能修正，和论文有区别的
    def get_buffer_zone_points(self):
        """得到叶子节点缓冲区内未被分配的点(索引)"""
        buffer_points_index_leaf = dict()
        B = self.get_neighbors()
        for nb in B:
            if  nb.is_allocated:
                continue
            for index in nb.indices:
                buffer_points_index_leaf[index] = nb
        return buffer_points_index_leaf

    def get_neighbors(self):
        """基于26相邻的相邻体素查找"""
        neighbors = self.find_leaf_face_nb()
        neighbors = neighbors + self.find_leaf_vertex_nb()
        neighbors = neighbors + self.find_leaf_edge_nb()
        return neighbors

    def draw_leaves(self):
        """以随机颜色显示叶子节点中的点"""
        np.random.seed(0)
        colors = [np.random.rand(3) for _ in range(len(self.root.leaves))]
        clouds = []
        for i,leaf in enumerate(self.root.leaves):
            points = []
            pcd = o3d.geometry.PointCloud()
            for p in leaf.indices:
                points.append(leaf.root.cloud[p])
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(colors[i])
            clouds.append(pcd)
        o3d.visualization.draw_geometries(clouds,width=1000,height=800,left=50,top=50)

    def find_leaf_face_nb(self):
        """找到面相邻的邻接体素"""
        faces = ['U','D','F','B','L','R']
        neighbors = []
        for face in faces:
            neighbor = self._get_face_ori(face)
            neighbors = neighbors + self._find_face_nb(neighbor, face)
        return neighbors

    def _get_face_ori(self,face):
        """在各个面方向上找到邻接大体素"""
        if face == 'U':
            if self.parent is None:
                return None
            if self.parent.children[4] == self:
                return self.parent.children[0]
            if self.parent.children[5] == self:
                return self.parent.children[1]
            if self.parent.children[6] == self:
                return self.parent.children[2]
            if self.parent.children[7] == self:
                return self.parent.children[3]
            node = self.parent._get_face_ori(face)
            if node is None or node.is_leaf:
                return node
            # leaf_node一定是一个上孩子
            if self.parent.children[0] == self:
                return node.children[4]
            elif self.parent.children[1] == self:
                return node.children[5]
            elif self.parent.children[2] == self:
                return node.children[6]
            else:
                return node.children[7]
        elif face == 'D':
            if self.parent is None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[4]
            if self.parent.children[1] == self:
                return self.parent.children[5]
            if self.parent.children[2] == self:
                return self.parent.children[6]
            if self.parent.children[3] == self:
                return self.parent.children[7]
            node = self.parent._get_face_ori(face)
            if node is None or node.is_leaf:
                return node
            # leaf_node一定是一个下孩子
            if self.parent.children[4] == self:
                return node.children[0]
            elif self.parent.children[5] == self:
                return node.children[1]
            elif self.parent.children[6] == self:
                return node.children[2]
            else:
                return node.children[3]
        elif face == 'F':
            if self.parent is None:
                return None
            if self.parent.children[1] == self:
                return self.parent.children[0]
            if self.parent.children[2] == self:
                return self.parent.children[3]
            if self.parent.children[5] == self:
                return self.parent.children[4]
            if self.parent.children[6] == self:
                return self.parent.children[7]
            node = self.parent._get_face_ori(face)
            if node is None or node.is_leaf:
                return node
            # leaf_node一定是一个前孩子
            if self.parent.children[0] == self:
                return node.children[1]
            elif self.parent.children[3] == self:
                return node.children[2]
            elif self.parent.children[4] == self:
                return node.children[5]
            else:
                return node.children[6]
        elif face == 'B':
            if self.parent is None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[1]
            if self.parent.children[3] == self:
                return self.parent.children[2]
            if self.parent.children[4] == self:
                return self.parent.children[5]
            if self.parent.children[7] == self:
                return self.parent.children[6]
            node = self.parent._get_face_ori(face)
            if node is None or node.is_leaf:
                return node
            # leaf_node一定是一个前孩子
            if self.parent.children[1] == self:
                return node.children[0]
            elif self.parent.children[2] == self:
                return node.children[3]
            elif self.parent.children[5] == self:
                return node.children[4]
            else:
                return node.children[7]
        elif face == 'L':
            if self.parent is None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[3]
            if self.parent.children[1] == self:
                return self.parent.children[2]
            if self.parent.children[4] == self:
                return self.parent.children[7]
            if self.parent.children[5] == self:
                return self.parent.children[6]
            node = self.parent._get_face_ori(face)
            if node is None or node.is_leaf:
                return node
            # leaf_node一定是一个左孩子
            if self.parent.children[3] == self:
                return node.children[0]
            elif self.parent.children[2] == self:
                return node.children[1]
            elif self.parent.children[6] == self:
                return node.children[5]
            else:
                return node.children[4]
        else:
            if self.parent is None:
                return None
            if self.parent.children[3] == self:
                return self.parent.children[0]
            if self.parent.children[2] == self:
                return self.parent.children[1]
            if self.parent.children[7] == self:
                return self.parent.children[4]
            if self.parent.children[6] == self:
                return self.parent.children[5]
            node = self.parent._get_face_ori(face)
            if node is None or node.is_leaf:
                return node
            # leaf_node一定是一个右孩子
            if self.parent.children[0] == self:
                return node.children[3]
            elif self.parent.children[1] == self:
                return node.children[2]
            elif self.parent.children[5] == self:
                return node.children[6]
            else:
                return node.children[7]

    def _find_face_nb(self, neighbor, face):
        """向树下搜索得到叶子"""
        candidates = [] if neighbor is None else [neighbor]
        neighbors = []
        while len(candidates) > 0:
            if candidates[0] is None:
                candidates.remove(candidates[0])
                continue
            elif candidates[0].is_leaf:
                neighbors.append(candidates[0])
            else:
                if face == 'U':
                    candidates.append(candidates[0].children[4])
                    candidates.append(candidates[0].children[5])
                    candidates.append(candidates[0].children[6])
                    candidates.append(candidates[0].children[7])
                elif face == 'D':
                    candidates.append(candidates[0].children[0])
                    candidates.append(candidates[0].children[1])
                    candidates.append(candidates[0].children[2])
                    candidates.append(candidates[0].children[3])
                elif face == 'F':
                    candidates.append(candidates[0].children[1])
                    candidates.append(candidates[0].children[2])
                    candidates.append(candidates[0].children[5])
                    candidates.append(candidates[0].children[6])
                elif face == 'B':
                    candidates.append(candidates[0].children[0])
                    candidates.append(candidates[0].children[3])
                    candidates.append(candidates[0].children[4])
                    candidates.append(candidates[0].children[7])
                elif face == 'L':
                    candidates.append(candidates[0].children[0])
                    candidates.append(candidates[0].children[1])
                    candidates.append(candidates[0].children[5])
                    candidates.append(candidates[0].children[4])
                else:
                    candidates.append(candidates[0].children[2])
                    candidates.append(candidates[0].children[3])
                    candidates.append(candidates[0].children[6])
                    candidates.append(candidates[0].children[7])
            candidates.remove(candidates[0])
        return neighbors

    def find_leaf_vertex_nb(self):
        """找到点相邻的邻接体素"""
        neighbors = []
        for i in range(8):
            neighbor = self._get_vertex_ori(i)
            neighbors = neighbors + self._find_vertex_nb(neighbor, i)
        return neighbors

    #在各个卦限方向上搜索顶点相邻体素
    def _get_vertex_ori(self, flag):
        """在各个flag代表卦限方向上找大体素"""
        if flag == 0:
            if self.parent is None:
                return None
            if self.parent.children[6] == self:
                return self.parent.children[0]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[6]
        elif flag == 1:
            if self.parent is None:
                return None
            if self.parent.children[7] == self:
                return self.parent.children[1]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[7]
        elif flag == 2:
            if self.parent is None:
                return None
            if self.parent.children[4] == self:
                return self.parent.children[2]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[4]
        elif flag == 3:
            if self.parent is None:
                return None
            if self.parent.children[5] == self:
                return self.parent.children[3]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[5]
        elif flag == 4:
            if self.parent is None:
                return None
            if self.parent.children[2] == self:
                return self.parent.children[4]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[2]
        elif flag == 5:
            if self.parent is None:
                return None
            if self.parent.children[3] == self:
                return self.parent.children[5]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[3]
        elif flag == 6:
            if self.parent is None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[6]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[0]
        else:
            if self.parent is None:
                return None
            if self.parent.children[1] == self:
                return self.parent.children[7]
            node = self.parent._get_vertex_ori(flag)
            if node is None or node.is_leaf:
                return node
            return node.children[1]

    def _find_vertex_nb(self, neighbor, flag):
        """向树下搜索得到叶子"""
        candidates = [] if neighbor is None else [neighbor]
        neighbors = []
        while len(candidates) > 0:
            if candidates[0] is None:
                candidates.remove(candidates[0])
                continue
            elif candidates[0].is_leaf:
                neighbors.append(candidates[0])
            else:
                if flag == 0:
                    candidates.append(candidates[0].children[6])
                elif flag == 1:
                    candidates.append(candidates[0].children[7])
                elif flag == 2:
                    candidates.append(candidates[0].children[4])
                elif flag == 3:
                    candidates.append(candidates[0].children[5])
                elif flag == 4:
                    candidates.append(candidates[0].children[2])
                elif flag == 5:
                    candidates.append(candidates[0].children[3])
                elif flag == 6:
                    candidates.append(candidates[0].children[0])
                else:
                    candidates.append(candidates[0].children[1])
            candidates.remove(candidates[0])
        return neighbors

    def find_leaf_edge_nb(self):
        """找到边相邻的邻接体素"""
        neighbors = []
        for i in range(12):
            neighbor = self._get_edge_ori(i+1)
            neighbors = neighbors + self._find_edge_nb(i+1, neighbor)
        return neighbors

    #以flag为依据在各个边方向上邻接体素查找
    def _get_edge_ori(self, flag):
        """向指定边方向搜索得到大体素"""
        if flag == 1:
            if self.parent == None:
                return None
            if self.parent.children[6] == self:
                return self.parent.children[1]
            if self.parent.children[7] == self:
                return self.parent.children[0]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[6] if self.parent.children[1] == self
                    else node.children[7])
        elif flag == 2:
            if self.parent == None:
                return None
            if self.parent.children[4] == self:
                return self.parent.children[1]
            if self.parent.children[7] == self:
                return self.parent.children[2]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[4] if self.parent.children[1] == self
                    else node.children[7])
        elif flag == 3:
            if self.parent == None:
                return None
            if self.parent.children[4] == self:
                return self.parent.children[3]
            if self.parent.children[5] == self:
                return self.parent.children[2]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[4] if self.parent.children[3] == self
                    else node.children[5])
        elif flag == 4:
            if self.parent == None:
                return None
            if self.parent.children[6] == self:
                return self.parent.children[3]
            if self.parent.children[5] == self:
                return self.parent.children[0]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[6] if self.parent.children[3] == self
                    else node.children[5])
        elif flag == 5:
            if self.parent == None:
                return None
            if self.parent.children[2] == self:
                return self.parent.children[0]
            if self.parent.children[6] == self:
                return self.parent.children[4]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[2] if self.parent.children[0] == self
                    else node.children[6])
        elif flag == 6:
            if self.parent == None:
                return None
            if self.parent.children[3] == self:
                return self.parent.children[1]
            if self.parent.children[7] == self:
                return self.parent.children[5]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[3] if self.parent.children[1] == self
                    else node.children[7])
        elif flag == 7:
            if self.parent == None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[2]
            if self.parent.children[4] == self:
                return self.parent.children[6]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[0] if self.parent.children[2] == self
                    else node.children[4])
        elif flag == 8:
            if self.parent == None:
                return None
            if self.parent.children[1] == self:
                return self.parent.children[3]
            if self.parent.children[5] == self:
                return self.parent.children[7]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[1] if self.parent.children[3] == self
                    else node.children[5])
        elif flag == 9:
            if self.parent == None:
                return None
            if self.parent.children[1] == self:
                return self.parent.children[4]
            if self.parent.children[2] == self:
                return self.parent.children[7]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[1] if self.parent.children[4] == self
                    else node.children[2])
        elif flag == 10:
            if self.parent == None:
                return None
            if self.parent.children[2] == self:
                return self.parent.children[5]
            if self.parent.children[3] == self:
                return self.parent.children[4]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[2] if self.parent.children[5] == self
                    else node.children[3])
        elif flag == 11:
            if self.parent == None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[5]
            if self.parent.children[3] == self:
                return self.parent.children[6]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[0] if self.parent.children[5] == self
                    else node.children[3])
        else:
            if self.parent == None:
                return None
            if self.parent.children[0] == self:
                return self.parent.children[7]
            if self.parent.children[1] == self:
                return self.parent.children[6]
            node = self.parent._get_edge_ori(flag)
            if node is None or node.is_leaf:
                return node
            return (node.children[0] if self.parent.children[7] == self
                    else node.children[1])


    def _find_edge_nb(self, flag, neighbor):
        """向树下搜索得到叶子"""
        candidates = [] if neighbor is None else [neighbor]
        neighbors = []
        while len(candidates) > 0:
            if candidates[0] is None:
                candidates.remove(candidates[0])
                continue
            elif candidates[0].is_leaf:
                neighbors.append(candidates[0])
            else:
                if flag == 1 :
                    candidates.append(candidates[0].children[6])
                    candidates.append(candidates[0].children[7])
                elif flag == 2:
                    candidates.append(candidates[0].children[4])
                    candidates.append(candidates[0].children[7])
                elif flag == 3:
                    candidates.append(candidates[0].children[4])
                    candidates.append(candidates[0].children[5])
                elif flag == 4:
                    candidates.append(candidates[0].children[5])
                    candidates.append(candidates[0].children[6])
                elif flag == 5:
                    candidates.append(candidates[0].children[2])
                    candidates.append(candidates[0].children[6])
                elif flag == 6:
                    candidates.append(candidates[0].children[3])
                    candidates.append(candidates[0].children[7])
                elif flag == 7:
                    candidates.append(candidates[0].children[0])
                    candidates.append(candidates[0].children[4])
                elif flag == 8:
                    candidates.append(candidates[0].children[1])
                    candidates.append(candidates[0].children[5])
                elif flag == 9:
                    candidates.append(candidates[0].children[1])
                    candidates.append(candidates[0].children[2])
                elif flag == 10:
                    candidates.append(candidates[0].children[2])
                    candidates.append(candidates[0].children[3])
                elif flag == 11:
                    candidates.append(candidates[0].children[0])
                    candidates.append(candidates[0].children[3])
                else:
                    candidates.append(candidates[0].children[0])
                    candidates.append(candidates[0].children[1])
            candidates.remove(candidates[0])
        return neighbors



if __name__ == '__main__':
    """测试代码"""
    points = np.loadtxt('data/chair/2.txt', dtype=float, usecols=(0, 1, 2)).tolist()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    oc = Octree(points, center=bb.get_center())
    settings = Settings(residual=0.02,res_th=0.02,dist_th=0.1)
    oc.create(settings)
    oc.draw_leaves()
