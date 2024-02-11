# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/2/2 10:19

from obrg import obrg_calculation
import numpy as np
from settings import Settings

if __name__ == '__main__':
    input_file = '' # 输入点云文件（txt）地址
    settings = Settings(residual=0.01,res_th=0.01,dist_th=0.05) # 阈值设定
    obrg_calculation(input_file, settings, draw=True, timing=True, evaluation=True) # 决定程序执行哪些操作