# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/1/23 10:35

from math import pi

class Settings:
    """存储算法中参数的一个类"""
    def __init__(self, residual = 6, res_th = 5, dist_th = 6.5):
        """初始化一些参数"""
        #构造八叉树所需的参数
        self.max_level = 8 #八叉树最大层数
        self.residual = residual #判断是否再进行体素化的残差阈值6
        self.lower_lim_num = 3 #叶子节点所能接受最少点的个数
        #区域生长算法所需要的参数
        self.res_th = res_th #用于判断是否可以作为种子点的残差阈值5
        self.ang_th = pi/18 #用于判断是否要合并体素的法向量角度阈值
        self.dist_th = dist_th #用于判断点离拟合平面是否足够近的阈值（6-7之间可以将每个体素考虑进去）
        self.min_segment = 100 #分割区域的最少点的个数