"""
HOI4D Semantic Segmentation Dataset for Pointcept.

Two sampling modes:
  sample_unit="clip" (default):
      Each sample = one 3-frame clip merged into a 24576-point cloud.
      100 non-overlapping clips per sequence -> 222100 train samples.

  sample_unit="sequence":
      Each sample = one full 300-frame sequence concatenated into a single
      ~2.46M-point cloud. 2221 train / 892 val samples.
      Returns clip_idx in {0..99} and frame_emb (frame index normalized to
      [-1, 1]); the LaCT-Temporal model chunks by clip_idx.

Splits:
  train: train2.h5 + train3.h5 + train4.h5  (2221 sequences)
  val:   test1_right.h5 + test2_right.h5    (892 sequences)

H5 layout per file:
  pcd:      (S, 300, 8192, 3) float64
  semantic: (S, 300, 8192) uint8

Author: Wentao
"""

import os
import numpy as np
import h5py

from .builder import DATASETS
from .defaults import DefaultDataset


_FILES = dict(
    train=["train2.h5", "train3.h5", "train4.h5"],  # train1.h5 not released
    val=["test1_right.h5", "test2_right.h5"],
)
_FRAMES_PER_CLIP = 3
_CLIPS_PER_SEQ = 100


@DATASETS.register_module()
class HOI4DDataset(DefaultDataset):
    """
    Args:
        sample_unit: "clip" returns one 3-frame clip per __getitem__ (default);
            "sequence" returns the full 300-frame sequence (for LaCT-Temporal).
    """

    def __init__(self, *args, sample_unit: str = "clip", **kwargs):
        assert sample_unit in ("clip", "sequence")
        self.sample_unit = sample_unit
        super().__init__(*args, **kwargs)
        self._h5_handles = None

    def get_data_list(self):
        self._files = _FILES[self.split]
        data_list = []
        for s, fn in enumerate(self._files):
            with h5py.File(os.path.join(self.data_root, fn), "r") as f:
                n = int(f["pcd"].shape[0])
            for d in range(n):
                if self.sample_unit == "clip":
                    for c in range(_CLIPS_PER_SEQ):
                        data_list.append((s, d, c))
                else:
                    data_list.append((s, d, None))
        return data_list

    def _ensure_open(self):
        if self._h5_handles is None:
            self._h5_handles = [
                h5py.File(os.path.join(self.data_root, fn), "r")
                for fn in self._files
            ]
            self._pcd = [h["pcd"] for h in self._h5_handles]
            self._semantic = [h["semantic"] for h in self._h5_handles]

    def get_data(self, idx):
        self._ensure_open()
        s, d, c = self.data_list[idx % len(self.data_list)]

        if self.sample_unit == "clip":
            frame0 = c * _FRAMES_PER_CLIP
            frame1 = frame0 + _FRAMES_PER_CLIP
            pcd = np.asarray(self._pcd[s][d, frame0:frame1], dtype=np.float32)
            sem = np.asarray(self._semantic[s][d, frame0:frame1], dtype=np.int32)
            return dict(coord=pcd.reshape(-1, 3), segment=sem.reshape(-1))

        pcd = np.asarray(self._pcd[s][d], dtype=np.float32)        # (300, 8192, 3)
        sem = np.asarray(self._semantic[s][d], dtype=np.int32)     # (300, 8192)
        T = pcd.shape[0]                                           # 300
        P = pcd.shape[1]                                           # 8192
        frame_idx = np.repeat(np.arange(T, dtype=np.int32), P)     # 0..299
        frame_emb = (frame_idx.astype(np.float32) / (T - 1)) * 2.0 - 1.0
        return dict(
            coord=pcd.reshape(-1, 3),
            segment=sem.reshape(-1),
            clip_idx=(frame_idx // _FRAMES_PER_CLIP).astype(np.int32),
            frame_emb=frame_emb.reshape(-1, 1),
        )
