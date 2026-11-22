# PoinTTT：Recurrent Point Transformer via Test-Time Training

[arXiv](https://arxiv.org/abs/2611.29059) · [Project page](https://pointtt.cs.virginia.edu/)

![PoinTTT](assets/teaser.png)

Initial release: two PoinTTT checkpoints and the code to evaluate them.

| Dataset | Test set | Metric | PoinTTT | Checkpoint |
|---|---|---|---|---|
| [FOR-instanceV2](https://doi.org/10.5281/zenodo.16742708) | 28 plots, all points | mIoU | **87.9** | `pointttt_forainetv2.pth` |
| [HOI4D](https://hoi4d.github.io/) (4D semantic segmentation) | 892 sequences, all points | pptr_mIoU | **42.6** | `pointttt_hoi4d.pth` |

FOR-instanceV2 in detail: mAcc 91.9, allAcc 96.0; IoU ground 99.3, wood 69.3, leaf 95.1.
pptr_mIoU is the mean IoU over the classes that occur in the HOI4D test labels, without classes 0 and 1
(39 classes; the protocol of PPTr), computed by `scripts/pptr_miou.py`.

## Installation

Linux with an NVIDIA GPU, conda (or mamba / micromamba) and gcc/g++ 9-13 (or `USE_CONDA_GCC=1`).
The script builds a conda environment in `./env` with Python 3.11, PyTorch, spconv, torch-scatter, flash-attn and
`libs/pointops`, then runs `scripts/smoke_test_env.py` (about 15 minutes; 8 GB, or 12 GB for torch 2.7.1).
Run ONE of the two commands:

```bash
git clone https://github.com/UVA-Computer-Vision-Lab/pointtt.git && cd pointtt
bash scripts/setup_env.sh                    # A100, H100, ...: torch 2.5.0+cu124
TORCH_VER=2.7.1 bash scripts/setup_env.sh    # or, on Blackwell GPUs (B200): torch 2.7.1+cu128, needs glibc >= 2.32
```

The scripts below use `./env/bin/python` when it exists; for an environment elsewhere set `PYTHON=/path/to/env/bin/python`.

## Data

Expected layout (other locations: set `DATA_ROOT`, see the scripts):

```
data/
├── forainetv2/
│   ├── meta_data/test_list.txt
│   └── test_data/*.ply
└── hoi4d/
    ├── test1_right.h5
    └── test2_right.h5
```

**FOR-instanceV2** ([Zenodo record 16742708](https://doi.org/10.5281/zenodo.16742708)). The script downloads
`test_data.zip` (377 MB), checks its md5, unpacks it and installs the official test list:

```bash
bash scripts/prepare_forest.sh
```

**HOI4D**. Download `test_wolabel_seg.h5` (52.6 GB) and `label.npy` (2.2 GB) of the semantic segmentation
challenge ("Sem Seg Test Set" on the [HOI4D page](https://hoi4d.github.io/); the training files are not needed)
into `data/hoi4d/SemSegChallenge/`, then split them into the two test files and check them:

```bash
sha256sum data/hoi4d/SemSegChallenge/*
# b78bf883f9bc1e6e7ec04b05bbd0095c225ec2abcf8b3db55ae8e28f81acc7ec  data/hoi4d/SemSegChallenge/test_wolabel_seg.h5
# 78af0ac5041b2577b5c66ad31663f20116015d344315cc082e2ef4f6d307bb17  data/hoi4d/SemSegChallenge/label.npy
HDF5_USE_FILE_LOCKING=FALSE ./env/bin/python pointcept/datasets/preprocessing/hoi4d/build_test_h5.py   # writes data/hoi4d/test{1,2}_right.h5 (55 GB)
bash scripts/check_hoi4d.sh
```

`data/hoi4d/SemSegChallenge/` can be deleted afterwards.

## Checkpoints

The checkpoints are attached to release [v1.0.0](https://github.com/UVA-Computer-Vision-Lab/pointtt/releases/tag/v1.0.0) of this
repository. Put them into `checkpoints/` and check them:

```bash
mkdir -p checkpoints
gh release download v1.0.0 --repo UVA-Computer-Vision-Lab/pointtt --pattern "pointttt_*.pth" --pattern SHA256SUMS --dir checkpoints
(cd checkpoints && sha256sum -c SHA256SUMS)
```

| File | Size | SHA-256 |
|---|---|---|
| `pointttt_forainetv2.pth` | 194 MB | `a82122fc110f786a07dd6785826a8d649d4be7fc84ebadffb07a66af25e5f7d8` |
| `pointttt_hoi4d.pth` | 193 MB | `ec1958f9560bc989653902a57778fd24eee688188f9d161d0f1e3d2bc56365b5` |

## Evaluation

**FOR-instanceV2** (1 GPU, a few minutes):

```bash
bash scripts/eval_forest.sh                 # checkpoints/pointttt_forainetv2.pth
```

Expected output (NVIDIA B200, torch 2.7.1+cu128):

```
Val result: mIoU/mAcc/allAcc 0.8789/0.9188/0.9601.
Class_0-ground Result: iou/accuracy 0.9928/0.9956
Class_1-wood Result: iou/accuracy 0.6929/0.7786
Class_2-leaf Result: iou/accuracy 0.9511/0.9820
```

**HOI4D** (1, 2 or 4 GPUs; about 75 minutes on 4 B200, several hours on one):

```bash
bash scripts/eval_hoi4d.sh                  # checkpoints/pointttt_hoi4d.pth
```

Expected last line (4 x NVIDIA B200, torch 2.7.1+cu128, `NUM_GPUS=4 SEQ_PER_GPU=8`):

```
[pptr_score] block=1 ts=... raw_mIoU=0.3388 pptr_mIoU=0.4257 active_classes=39 raw_mAcc=0.4442 raw_allAcc=0.7803
```

`raw_mIoU` is the mean over all 49 classes, as printed by the evaluator; the reported number is `pptr_mIoU`.
By default the script uses 4, 2 or 1 GPUs (the most that are visible) and 8 sequences per GPU, which needs about
18 GB of GPU memory for the batch buffers alone; on smaller GPUs try `SEQ_PER_GPU=2` to `4`. Other settings
change the random parts of the evaluation (voxel representatives, serialization order) and move the result by
about 0.0005 (pptr_mIoU 0.425-0.426, i.e. 42.5-42.6). The reference run used 24 CPU cores and 184 GB of host
memory for 4 GPUs. The data loaders pass batches through shared memory (about 2 GB for FOR-instanceV2, about 7 GB
per GPU for HOI4D); in containers use `--ipc=host` or a large `--shm-size`.

Logs go to `exp/forainetv2_eval/train.log` and `exp/hoi4d_eval/train.log` (`OUT=...` to change). The checkpoint
is loaded strictly, so a wrong file stops the run with an error; compare the SHA-256 above to detect a damaged
download. Small differences in the last digit are possible on other GPU types or library versions.

## License

The code is released under the MIT License (`LICENSE`); third-party code keeps its own notices
(`THIRD_PARTY_NOTICES.md`). The datasets are not redistributed here and have their own licences:
FOR-instanceV2 is released under GPL-3.0-or-later and includes plots of
[FOR-instance](https://doi.org/10.5281/zenodo.8287792) (CC BY 4.0); HOI4D is released under CC BY-NC 4.0.
Each checkpoint was trained on the named dataset and is subject to its terms; the HOI4D checkpoint is for
non-commercial research use only.

## Acknowledgements

This code builds on [Pointcept](https://github.com/Pointcept/Pointcept) and Point Transformer V3, and on
[LaCT](https://github.com/a1600012888/LaCT) for the fast-weight layers. We thank the authors of
[FOR-instanceV2 / ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D), FOR-instance and
[HOI4D](https://hoi4d.github.io/) for releasing their data.

## Citation

```bibtex
@article{zhou2026recurrent,
  title   = {Recurrent Point Transformer via Test-Time Training},
  author  = {Zhou, Wentao and Chen, Xuweiyi and Saunders, Charles and RoyChowdhury, Aruni and Cheng, Zezhou},
  journal = {arXiv preprint arXiv:2611.29059},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.29059}
}
```
