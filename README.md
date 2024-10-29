# Octree-based-Region-Growing
This is an undergraduate graduation design project, the purpose is to get familiar with point cloud segmentation and lay a foundation for further study in this field.
Our project is based on [this work](https://github.com/lupeterm/OBRG-Py), which reproduces the basic ideas of the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271615000283) very well. Our code has been updated in terms of flatness detection, refinement process, etc., and a 26-neighbor voxel search algorithm has been introduced.

# Table of Contents
- [Execution Specification](#ExecutionSpecification) 
- [Performance](#Performance)
  - [Visualization](##Visualization)
  - [Evaluation](##Evaluation)
- [Reference](#Reference)


# Execution Specification
By setting the point cloud file path and parameters(residuals, visualization, evaluation, timing), we can directly run the main.py file to achieve point cloud segmentation.
```sh
input_file = '' # 输入点云文件（txt）地址
settings = Settings(residual=0.01,res_th=0.01,dist_th=0.05) # 阈值设定
obrg_calculation(input_file, settings, draw=True, timing=True, evaluation=True) # 决定程序执行哪些操作
```
# Performance
## Visualization
![error](https://github.com/kangjie-ding/Octree-based-Region-Growing/blob/main/test_data/visualization/visualization.jpg "visualization of our algorithm compared to traditional methods")
## Evaluation
![error](https://github.com/kangjie-ding/Octree-based-Region-Growing/blob/main/test_data/visualization/evaluation.jpg "evaluation of our algorithm compared to traditional methods")

# Reference
[Octree-based region growing](https://www.sciencedirect.com/science/article/abs/pii/S0924271615000283)
