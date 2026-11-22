"""
ForAINetV2 (FOR-instanceV2) Forest LiDAR Dataset.

Reads .ply files directly — no preprocessing needed.
Each ply contains up to millions of points with fields:
  x, y, z (float32) — coordinates
  semantic_seg (int32) — class label (1 ground, 2 wood, 3 leaf; 0 unlabeled)
  treeID (int32) — instance ID

Splits are read from meta_data/{train,val,test}_list.txt.

Author: Wentao
"""

import os
import numpy as np
import plyfile

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ForAINetV2Dataset(DefaultDataset):
    """
    data_root should point to the ForAINetV2 root containing:
      train_val_data/*.ply
      test_data/*.ply
      meta_data/{train,val,test}_list.txt

    Args:
        ignore_unannotated_tree (bool): if True (default), wood / leaf points
            with treeID == 0 get semantic label = ignore. The plots of the
            official train / val / test lists contain no such point, so there
            the switch changes no label.
    """

    def __init__(
        self,
        *args,
        ignore_unannotated_tree: bool = True,
        **kwargs,
    ):
        self.ignore_unannotated_tree = ignore_unannotated_tree
        super().__init__(*args, **kwargs)

    def get_data_list(self):
        """Read split list from meta_data/{split}_list.txt."""
        meta_root = os.path.join(self.data_root, "meta_data")

        if isinstance(self.split, str):
            splits = [self.split]
        else:
            splits = list(self.split)

        data_list = []
        for split in splits:
            list_file = os.path.join(meta_root, f"{split}_list.txt")
            with open(list_file, "r") as f:
                names = [line.strip() for line in f if line.strip()]

            # Files are in train_val_data/ for train+val, test_data/ for test
            subdir = "test_data" if split == "test" else "train_val_data"
            for name in names:
                ply_path = os.path.join(self.data_root, subdir, f"{name}.ply")
                if os.path.exists(ply_path):
                    data_list.append(ply_path)
                else:
                    # Some files might not exist — log and skip
                    print(f"[ForAINetV2] missing: {ply_path}")
        return data_list

    def get_data(self, idx):
        ply_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        plydata = plyfile.PlyData.read(ply_path)
        v = plydata["vertex"]
        coord = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

        if "semantic_seg" in v.data.dtype.names:
            # Raw labels are 1-indexed {1, 2, 3}; 0 = unlabeled.
            # Map to 0-indexed {0, 1, 2}, with 0/unlabeled → ignore_index.
            raw = np.asarray(v["semantic_seg"], dtype=np.int32).reshape(-1)
            segment = raw - 1
            segment[raw == 0] = self.ignore_index
        else:
            segment = np.full(coord.shape[0], self.ignore_index, dtype=np.int32)

        if "treeID" in v.data.dtype.names:
            # Raw treeID convention in ForAINetV2:
            #   0 = unannotated (point has no valid instance)
            #   >0 = valid tree instance ID (1-indexed)
            # Remap to pointcept convention:
            #   -1 = ignore (unannotated OR on a segment_ignore_index class)
            #   >=0 = raw treeID passed through (InstanceParser will
            #         re-compact these to a contiguous 0..N-1 range later).
            raw = np.asarray(v["treeID"], dtype=np.int32).reshape(-1)
            instance = raw.copy()
            instance[raw == 0] = -1                 # unannotated
            # Ground (semantic 0) and unlabeled (-1) points never belong to a
            # tree instance — force to ignore even if treeID accidentally set.
            instance[segment == 0] = -1
            instance[segment == self.ignore_index] = -1
            # Tree-class points (wood=1 / leaf=2) without a valid treeID would
            # survive InstanceParser's `~in1d(segment, ignore)` mask and get
            # re-mapped into a phantom instance id. Mark their semantic as
            # ignore so they are cleanly excluded from instance construction.
            unannotated_tree = (raw == 0) & (segment != self.ignore_index) & (segment != 0)
            if self.ignore_unannotated_tree:
                segment[unannotated_tree] = self.ignore_index
        else:
            instance = np.full(coord.shape[0], -1, dtype=np.int32)

        return dict(
            coord=coord,
            segment=segment,
            instance=instance,
            name=name,
            condition="ForAINet",
        )

    def get_data_name(self, idx):
        ply_path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(ply_path))[0]
