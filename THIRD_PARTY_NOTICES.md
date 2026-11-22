# Third-party notices

The code in this repository is released under the MIT License (see `LICENSE`). It is built on
[Pointcept](https://github.com/Pointcept/Pointcept) (MIT License, Copyright (c) 2023 Pointcept), and some
files contain code adapted from the projects listed below. Those files keep the copyright and licence
notices of their origin.

## Code

| Project | Licence and copyright | Used in |
|---|---|---|
| [LaCT](https://github.com/a1600012888/LaCT) | MIT, Copyright (c) 2025 Tianyuan Zhang, Hao Tan | `pointcept/models/point_transformer_v3/lact_layer.py` (copied); fast-weight update of the TTT layers in `point_transformer_v3_lact_dec.py` and `point_transformer_v3_lact_temporal.py` (adapted) |
| [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | MIT, Copyright (c) 2024 Keller Jordan | Newton-Schulz iteration in `lact_layer.py` (via LaCT) |
| [Moonlight](https://github.com/MoonshotAI/Moonlight) | MIT, Copyright (c) 2025 Moonshot AI | Newton-Schulz iteration in `lact_layer.py` (via LaCT) |
| [detectron2](https://github.com/facebookresearch/detectron2) / [fvcore](https://github.com/facebookresearch/fvcore) | Apache-2.0, Copyright (c) Facebook, Inc. and its affiliates | `pointcept/utils/comm.py`, `pointcept/utils/events.py`, `pointcept/engines/launch.py`, `pointcept/engines/defaults.py`, `pointcept/engines/train.py` (TrainerBase), `pointcept/engines/hooks/default.py` |
| [mmcv](https://github.com/open-mmlab/mmcv) | Apache-2.0, Copyright (c) OpenMMLab | `pointcept/utils/config.py`, `pointcept/utils/registry.py`, `pointcept/utils/path.py`, `pointcept/utils/logger.py`, `pointcept/utils/misc.py` (`is_seq_of`, `is_str`, `import_modules_from_strings`) |
| [O-CNN](https://github.com/octree-nn/ocnn-pytorch) | MIT, Copyright (c) 2022 Peng-Shuai Wang | `pointcept/models/utils/serialization/z_order.py` |
| [numpy-hilbert-curve](https://github.com/PrincetonLIPS/numpy-hilbert-curve) | MIT, Copyright (c) 2020 Princeton Laboratory for Intelligent Probabilistic Systems | `pointcept/models/utils/serialization/hilbert.py` |
| [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt), [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax) | MIT, Copyright (c) 2019 Pavel Iakubovskii; Copyright (c) Eugene Khvedchenya; Copyright (c) 2018 Maxim Berman | `pointcept/models/losses/lovasz.py` |
| [semseg](https://github.com/hszhao/semseg) | MIT, Copyright (c) 2019 Hengshuang Zhao | `pointcept/utils/misc.py` (`intersection_and_union`, `intersection_and_union_gpu`) |
| [PointWeb](https://github.com/hszhao/PointWeb) | MIT, Copyright (c) 2019 Hengshuang Zhao | `libs/pointops` |
| [SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation) | MIT, Copyright (c) Chris Choy | `pointcept/datasets/transform.py` |

The Apache License 2.0 is reproduced in `licenses/Apache-2.0.txt`. Files derived from Apache-2.0 code that were
changed for this repository say so in their header.

Each MIT-licensed project above is distributed under the following terms, with its own copyright notice:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data files

| File | Origin | Licence |
|---|---|---|
| `pointcept/datasets/preprocessing/forainetv2/meta_data/test_list.txt` | [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D), `data/ForAINetV2/meta_data/test_list.txt` (commit 6a75c37), unchanged | CC BY-NC 4.0 |

The datasets themselves are not part of this repository; see the README for their sources and licences.
