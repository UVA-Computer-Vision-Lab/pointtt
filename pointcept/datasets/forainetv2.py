"""
ForAINetV2 (FOR-instanceV2) Forest LiDAR Dataset.

Reads the .ply files directly. Each ply holds the fields
  x, y, z (float32)        coordinates
  semantic_seg (int32)     class label: 1 ground, 2 wood, 3 leaf (0 unlabeled)
  treeID (int32)           tree instance ID

Splits are read from meta_data/{split}_list.txt; the plots are in test_data/ for the
test split and in train_val_data/ otherwise.

Author: Wentao
"""

import os
import numpy as np
import plyfile

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ForAINetV2Dataset(DefaultDataset):
    def get_data_list(self):
        list_file = os.path.join(self.data_root, "meta_data", f"{self.split}_list.txt")
        with open(list_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        subdir = "test_data" if self.split == "test" else "train_val_data"
        return [os.path.join(self.data_root, subdir, f"{name}.ply") for name in names]

    def get_data(self, idx):
        v = plyfile.PlyData.read(self.data_list[idx % len(self.data_list)])["vertex"]
        coord = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        # 1 ground, 2 wood, 3 leaf -> 0, 1, 2; 0 (unlabeled) -> -1 (ignored)
        segment = np.asarray(v["semantic_seg"], dtype=np.int32).reshape(-1) - 1
        instance = np.asarray(v["treeID"], dtype=np.int32).reshape(-1)
        return dict(coord=coord, segment=segment, instance=instance)
