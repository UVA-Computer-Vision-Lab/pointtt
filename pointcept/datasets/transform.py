"""
3D Point Cloud Augmentation

Inspirited by chrischoy/SpatioTemporalSegmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numpy as np
import torch
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict):
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[value] = data_dict[key].copy()
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, Mapping):
            return {sub_key: self(item) for sub_key, item in data.items()}
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        x_min, y_min, z_min = data_dict["coord"].min(axis=0)
        x_max, y_max, _ = data_dict["coord"].max(axis=0)
        if self.apply_z:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        else:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.p = p
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if self.center is None:
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, z_max = data_dict["coord"].max(axis=0)
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        else:
            center = self.center
        data_dict["coord"] -= center
        data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
        data_dict["coord"] += center
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None):
        self.scale = scale if scale is not None else [0.95, 1.05]

    def __call__(self, data_dict):
        scale = np.random.uniform(self.scale[0], self.scale[1], 1)
        data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
        if np.random.rand() < self.p:
            data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        jitter = np.clip(
            self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
            -self.clip,
            self.clip,
        )
        data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    """Voxelize: keep one random point per grid cell (FNV-1a hash of the cell)."""

    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "segment"),
        return_grid_coord=True,
    ):
        assert hash_type == "fnv" and mode == "train" and return_grid_coord
        self.grid_size = grid_size
        self.keys = keys

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        key = self.fnv_hash_vec(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, count = np.unique(key_sort, return_counts=True)
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        data_dict["grid_coord"] = grid_coord[idx_unique]
        for key in self.keys:
            data_dict[key] = data_dict[key][idx_unique]
        return data_dict

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
