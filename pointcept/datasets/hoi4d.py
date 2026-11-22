"""
HOI4D Semantic Segmentation Dataset for Pointcept.

Supports three sampling modes:
  sample_unit="clip" (default):
      Each sample = one 3-frame clip merged into a 24576-point cloud.
      100 non-overlapping clips per sequence → 222100 train / 89200 val samples.
      Returns frame_idx ∈ {0,1,2} and frame_emb (normalized to [-1,1]).

  sample_unit="sequence":
      Each sample = one full 300-frame sequence concatenated into a single
      ~2.46M-point cloud. 2221 train / 892 val samples.
      Returns frame_idx ∈ {0..299}, clip_idx ∈ {0..99}, and frame_emb.
      Intended for the LaCT-Temporal model that chunks by clip_idx.

  sample_unit="subseq":
      Each sample = a sliding window of `clips_per_sample` consecutive clips
      (window-length = clips_per_sample * frames_per_clip frames). Windows
      advance by `subseq_stride` clips (defaults to non-overlapping). clip_idx
      and frame_idx are remapped to window-local 0-based indices so the
      LaCT-Temporal model sees a self-contained K-clip sequence regardless of
      where the window sits in the full 100-clip recording. Lets you dial the
      effective TTT sequence length (e.g. 8 clips = 24 frames) instead of the
      full 300-frame sequence. With clips_per_sample=8 → 13 windows/seq (last
      is the 4-clip remainder), so 2221*13 train / 892*13 val samples.

Splits:
  train: train2.h5 + train3.h5 + train4.h5  (2221 sequences)
  val:   test1_right.h5 + test2_right.h5    (892 sequences)
  test:  test1_right.h5 + test2_right.h5    (same as val)

H5 layout per file:
  pcd:      (S, 300, 8192, 3) float64
  center:   (S, 300, 3) float64
  semantic: (S, 300, 8192) uint8

Author: Wentao
"""

import os
import numpy as np
import h5py

from .builder import DATASETS
from .defaults import DefaultDataset


_TRAIN_FILES = ["train2.h5", "train3.h5", "train4.h5"]   # train1.h5 not released
_TRAIN_SHARD_SIZES = [750, 750, 721]                     # sums to 2221
_TEST_FILES = ["test1_right.h5", "test2_right.h5"]
_TEST_SHARD_SIZES = [500, 392]                           # sums to 892
_FRAMES_PER_CLIP = 3
_FRAMES_PER_SEQ = 300
_POINTS_PER_FRAME = 8192


@DATASETS.register_module()
class HOI4DDataset(DefaultDataset):
    """
    Args:
        sample_unit: "clip" returns one 3-frame clip per __getitem__ (default);
            "sequence" returns the full 300-frame sequence (for LaCT-Temporal);
            "subseq" returns a window of `clips_per_sample` consecutive clips.
        frames_per_clip: frames merged per clip (also the clip granularity used
            by "sequence"/"subseq" when assigning clip_idx).
        clips_per_sample: REQUIRED for sample_unit="subseq" — number of clips
            per window (window = clips_per_sample * frames_per_clip frames).
        subseq_stride: clip-stride between consecutive windows (subseq only);
            defaults to clips_per_sample (non-overlapping windows).
    """

    def __init__(
        self,
        *args,
        sample_unit: str = "clip",
        frames_per_clip: int = _FRAMES_PER_CLIP,
        clips_per_sample: int = None,
        subseq_stride: int = None,
        **kwargs,
    ):
        assert sample_unit in ("clip", "sequence", "subseq")
        assert _FRAMES_PER_SEQ % frames_per_clip == 0, (
            f"frames_per_clip={frames_per_clip} must divide {_FRAMES_PER_SEQ}"
        )
        self.sample_unit = sample_unit
        self.frames_per_clip = frames_per_clip
        self.clips_per_seq = _FRAMES_PER_SEQ // frames_per_clip
        if sample_unit == "subseq":
            assert clips_per_sample is not None and clips_per_sample > 0, (
                "sample_unit='subseq' requires clips_per_sample > 0"
            )
            self.clips_per_sample = clips_per_sample
            self.subseq_stride = (
                subseq_stride if subseq_stride is not None else clips_per_sample
            )
        else:
            self.clips_per_sample = clips_per_sample
            self.subseq_stride = subseq_stride
        super().__init__(*args, **kwargs)
        self._h5_handles = None

    def get_data_list(self):
        splits = [self.split] if isinstance(self.split, str) else list(self.split)

        if any(s == "train" for s in splits):
            files, shards = _TRAIN_FILES, _TRAIN_SHARD_SIZES
        elif any(s in ("val", "test") for s in splits):
            files, shards = _TEST_FILES, _TEST_SHARD_SIZES
        else:
            raise ValueError(f"Unknown split {splits} for HOI4DDataset")

        # Take the number of sequences from the files themselves, so a
        # partial download or a small test file fails loudly / just works
        # instead of indexing past the end of a shard. A shard that is not
        # there keeps the official count: the val set of a training-only
        # config is built but never read.
        actual = []
        for fn, expected in zip(files, shards):
            path = os.path.join(self.data_root, fn)
            if not os.path.isfile(path):
                actual.append(expected)
                continue
            with h5py.File(path, "r") as f:
                n = int(f["pcd"].shape[0])
            if n != expected:
                print(f"[HOI4D] {fn}: {n} sequences on disk (official release has {expected})")
            actual.append(n)
        shards = actual

        self._files = files
        self._shard_sizes = shards
        data_list = []
        if self.sample_unit == "clip":
            for s, n in enumerate(shards):
                for d in range(n):
                    for c in range(self.clips_per_seq):
                        data_list.append((s, d, c))
        elif self.sample_unit == "subseq":
            # Window start clips: 0, stride, 2*stride, ... < clips_per_seq.
            for s, n in enumerate(shards):
                for d in range(n):
                    for c0 in range(0, self.clips_per_seq, self.subseq_stride):
                        data_list.append((s, d, c0))
        else:  # "sequence"
            for s, n in enumerate(shards):
                for d in range(n):
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

    def _make_frame_emb(self, frame_idx_array, max_frames):
        """Encode integer frame index as a normalized [-1, 1] float per point.
        Shape (N, 1)."""
        if max_frames <= 1:
            return np.zeros((frame_idx_array.shape[0], 1), dtype=np.float32)
        emb = (frame_idx_array.astype(np.float32) / (max_frames - 1)) * 2.0 - 1.0
        return emb.reshape(-1, 1)

    def get_data(self, idx):
        self._ensure_open()
        s, d, c = self.data_list[idx % len(self.data_list)]

        if self.sample_unit == "clip":
            frame0 = c * self.frames_per_clip
            frame1 = frame0 + self.frames_per_clip
            pcd = np.asarray(self._pcd[s][d, frame0:frame1], dtype=np.float32)
            sem = np.asarray(self._semantic[s][d, frame0:frame1], dtype=np.int32)
            T = self.frames_per_clip
            P = pcd.shape[1]
            coord = pcd.reshape(-1, 3)
            segment = sem.reshape(-1)
            frame_idx = np.repeat(np.arange(T, dtype=np.int32), P)
            clip_idx = np.full(coord.shape[0], c, dtype=np.int32)
            frame_emb = self._make_frame_emb(frame_idx, T)
        elif self.sample_unit == "subseq":
            # Window = clips [c0, c1); remap clip_idx/frame_idx to window-local
            # 0-based so the model sees a self-contained K-clip sequence.
            c0 = c
            c1 = min(c0 + self.clips_per_sample, self.clips_per_seq)
            frame0 = c0 * self.frames_per_clip
            frame1 = c1 * self.frames_per_clip
            pcd = np.asarray(self._pcd[s][d, frame0:frame1], dtype=np.float32)
            sem = np.asarray(self._semantic[s][d, frame0:frame1], dtype=np.int32)
            T = frame1 - frame0                                        # window frames
            P = pcd.shape[1]
            coord = pcd.reshape(-1, 3)
            segment = sem.reshape(-1)
            frame_idx = np.repeat(np.arange(T, dtype=np.int32), P)     # 0..T-1 local
            clip_idx = (frame_idx // self.frames_per_clip).astype(np.int32)  # 0..K-1 local
            frame_emb = self._make_frame_emb(frame_idx, T)
        else:  # "sequence"
            pcd = np.asarray(self._pcd[s][d], dtype=np.float32)        # (300, 8192, 3)
            sem = np.asarray(self._semantic[s][d], dtype=np.int32)     # (300, 8192)
            T = pcd.shape[0]                                           # 300
            P = pcd.shape[1]                                           # 8192
            coord = pcd.reshape(-1, 3)                                 # (300*8192, 3)
            segment = sem.reshape(-1)
            frame_idx = np.repeat(np.arange(T, dtype=np.int32), P)     # 0..299
            clip_idx = (frame_idx // self.frames_per_clip).astype(np.int32)
            frame_emb = self._make_frame_emb(frame_idx, T)

        return dict(
            coord=coord,
            segment=segment,
            frame_idx=frame_idx,
            clip_idx=clip_idx,
            frame_emb=frame_emb,
            name=self.get_data_name(idx),
            condition="HOI4D",
        )

    def get_data_name(self, idx):
        s, d, c = self.data_list[idx % len(self.data_list)]
        base = self._files[s].replace(".h5", "")
        if self.sample_unit == "sequence":
            return f"{base}_seq{d:04d}"
        if self.sample_unit == "subseq":
            return f"{base}_seq{d:04d}_win{c:03d}"
        return f"{base}_seq{d:04d}_clip{c:03d}"
