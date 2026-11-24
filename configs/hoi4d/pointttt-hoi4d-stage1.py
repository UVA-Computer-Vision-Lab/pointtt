"""PoinTTT on HOI4D, stage 1: PTv3 on independent 3-frame clips (24,576 points each).

8 GPUs x 8 clips (batch 64), AMP, 3,470 iterations per epoch over the 222,100 clips of
train2-4.h5 (2,221 sequences), no evaluation during training; every epoch is saved
(model/epoch_N.pth, about 550 MB each). The schedule has 50 epochs, but stage 2
(pointttt-hoi4d-stage2.py) starts from model/epoch_48.pth: the run behind the released
checkpoint was stopped after epoch 48.

    bash scripts/train.sh -d hoi4d -c pointttt-hoi4d-stage1 -g 8 -n exp/hoi4d/stage1

About 24 minutes per epoch on 8 NVIDIA B200, about 62 GB of GPU memory per GPU at 8 clips per
GPU (nvidia-smi peak in the first minutes on a B200). The total batch of 64 clips belongs to the
learning rate and to the 3,470 iterations per epoch; with fewer GPUs keep it, e.g. -g 4 (16 clips
per GPU, over 100 GB of GPU memory each), or replay the 8-GPU run in one process:

    bash scripts/train.sh -d hoi4d -c pointttt-hoi4d-stage1 -g 1 -n exp/hoi4d/stage1 \
        -o "batch_size=8 num_worker=4 grad_accum_steps=8 ddp_replay_world_size=8"

Replay mode (pointcept/engines/train.py, ReplayRanksSampler) draws the batches of all 8 ranks in
turn and makes one optimizer step per 8 of them, so every step sees the clips of one step of the
8-GPU run; it takes about 8 times as long. GPU kernels are not deterministic, so no two runs give
identical weights. The run behind the released checkpoint used this config and seed, but not one
setting throughout: epochs 1-8 in replay mode on 1 GPU, 9-30 on 4 GPUs x 16 clips, 31-48 on
8 GPUs x 8 clips (each part continued with -r true from the newest epoch checkpoint, with the AMP
loss scale carried over unchanged).
"""

_base_ = ["../_base_/default_runtime.py"]

# runtime
seed = 51329219
num_worker = 32  # loader workers in total over all GPUs
batch_size = 64  # clips in total over all GPUs
epoch = 50
eval_epoch = 50
evaluate = False
enable_amp = True
empty_cache = False
mix_prob = 0.0
param_dicts = [dict(keyword="block", lr=6e-4)]
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=10),
]
train = dict(type="DefaultTrainer")

# model
model = dict(
    type="DefaultSegmentorV2",
    num_classes=49,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# optimizer
optimizer = dict(type="AdamW", lr=6e-3, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[6e-3, 6e-4],  # [all other parameters, transformer blocks]
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# data
data_root = "data/hoi4d"  # train2.h5, train3.h5, train4.h5 (see README)
data = dict(
    train=dict(
        type="HOI4DDataset",
        split="train",  # train2-4.h5; sample_unit "clip" (default): 100 clips per sequence
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord",)),
        ],
    ),
)
