# working institution:School of Mathematical Sciences,Zhejiang University
# author:Kangjie Ding
# date:2023/2/19 15:19

def partition_judegement(point, node) -> int:
    """通过将点与当前节点中点比较，得出划分索引"""
    if (point[0] >= node.center[0]) and (point[1] >= node.center[1]) and (point[2] >= node.center[2]):
        index = 0
    elif (point[0] <= node.center[0]) and (point[1] >= node.center[1]) and (point[2] >= node.center[2]):
        index = 1
    elif (point[0] <= node.center[0]) and (point[1] <= node.center[1]) and (point[2] >= node.center[2]):
        index = 2
    elif (point[0] >= node.center[0]) and (point[1] <= node.center[1]) and (point[2] >= node.center[2]):
        index = 3
    elif (point[0] >= node.center[0]) and (point[1] >= node.center[1]) and (point[2] <= node.center[2]):
        index = 4
    elif (point[0] <= node.center[0]) and (point[1] >= node.center[1]) and (point[2] <= node.center[2]):
        index = 5
    elif (point[0] <= node.center[0]) and (point[1] <= node.center[1]) and (point[2] <= node.center[2]):
        index = 6
    else:
        index = 7
    return index