"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
