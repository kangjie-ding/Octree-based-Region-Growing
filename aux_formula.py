# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/1/23 10:49

import math
import numpy as np


def unit_vector(n):
    """返回单位化向量"""
    return n / np.linalg.norm(n)



def ang_div(n1, n2):
    """计算两个向量之间的夹角"""
    v1_u = unit_vector(n1)
    v2_u = unit_vector(n2)
    return np.arccos(np.clip(np.inner(v1_u, v2_u), -1.0, 1.0))

def dist(point, normal, para_d):
    """计算point到经过点para_d，以normal为法向量的平面的距离"""
    d = abs(normal[0]*point[0] + normal[1]*point[1] + normal[2] *
            point[2] - para_d)/math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    return d

def cal_evaluation(complete_segs, reference_segs, reference_data):
    """输入算法得到的分割结果和参考分割结果，计算得到precision,recall,F1-score和mIoU"""
    #首先将输入complete_segs转换为reference_segs相同格式
    tmp_segs = []
    for seg in complete_segs:
        segment = set()
        indices = []
        for leaf in seg:
            for index in leaf.indices:
                indices.append(index)
        for index in set(indices):
            segment.add(tuple(reference_data[index][0:3]))
        tmp_segs.append(segment)
    complete_segs = tmp_segs
    #计算TP,FP，FN
    TP_FP = 0
    TP_FN = 0
    TP = 0
    tmp_mIoU = 0
    for segment1 in reference_segs:
        TP_FN += len(segment1)
        current_tp = 0
        current_index = 0
        for index,segment2 in enumerate(complete_segs):
            if len(segment1.intersection(segment2)) > current_tp:
                current_tp = len(segment1.intersection(segment2))
                current_index = index
        TP += current_tp
        TP_FP += len(complete_segs[current_index])
        tmp_mIoU += current_tp/len(segment1.union(complete_segs[current_index]))
    #计算precision,recall和F1-score
    precision = TP/(TP_FP)
    recall = TP/(TP_FN)
    F1_score = 2*precision*recall/(precision+recall)
    mIoU = tmp_mIoU/len(reference_segs)
    return precision,recall,F1_score,mIoU
