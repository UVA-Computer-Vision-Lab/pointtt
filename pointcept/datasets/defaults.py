"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

Modified for PoinTTT: single-threaded OpenBLAS in the data loader workers; only the
part of DefaultDataset that the datasets of this repository use is kept.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger

from .transform import Compose


class DefaultDataset(Dataset):
    """Base class of the datasets: subclasses implement get_data_list() and get_data(idx)."""

    def __init__(self, split="train", data_root="data/dataset", transform=None, loop=1):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))

    def __len__(self):
        return len(self.data_list) * self.loop
