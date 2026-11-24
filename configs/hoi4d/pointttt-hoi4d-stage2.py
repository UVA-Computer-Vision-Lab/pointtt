"""PoinTTT on HOI4D, stage 2: the stage-1 PTv3 plus 8 LaCT (TTT) layers, fine-tuned on full
300-frame sequences (100 chunks of 3 frames, fast weights carried from chunk to chunk).

Starts from model/epoch_48.pth of stage 1 (pointttt-hoi4d-stage1.py); the 88 TTT keys that it lacks
start from their seeded initialisation (logged as missing keys). Every epoch is saved
(model/epoch_N.pth, about 0.55 GB each). The schedule has 50 epochs; the run behind the released
checkpoint was stopped after epoch 48.

    bash scripts/train.sh -d hoi4d -c pointttt-hoi4d-stage2 -g 1 -n exp/hoi4d/stage2 \
        -w exp/hoi4d/stage1/model/epoch_48.pth

One GPU, 4 sequences per iteration, emulating rank 0 of an 8-process run (ddp_emulate_world_size):
each epoch it trains on rank 0's share of the sequences (69 iterations). The model runs every
backward pass inside its forward pass, chunk by chunk, where DistributedDataParallel does not
average gradients, so the 8 processes of a multi-GPU launch train independently and only rank 0 is
saved; one process fed with rank 0's data is the same training at 1/8 of the GPU time. AMP
autocast is on; the per-chunk backward passes run outside the GradScaler, as in the original run.

Time: about 50 minutes per epoch on one NVIDIA B200 for the release run, which used a batched
version of the fast-weight update (same results up to rounding). This repository updates the
sequences of a batch one after the other, about 1.15 times slower per iteration (63-66 s instead
of 56-57 s for the first iterations on a B200): roughly 2 days to epoch 48. Peak GPU memory about
32 GB (nvidia-smi).
"""

_base_ = ["../_base_/default_runtime.py"]

# runtime
weight = "exp/hoi4d/stage1/model/epoch_48.pth"  # stage 1; scripts/train.sh -w overrides it
seed = 43366092
num_worker = 2  # loader workers in total over all GPUs
batch_size = 4  # sequences in total over all GPUs, at most max_batch_size_per_gpu per GPU
ddp_emulate_world_size = 8  # one process = rank 0 of an 8-process run; ignored with more than 1 GPU
epoch = 50
eval_epoch = 50
evaluate = False  # evaluate checkpoints with scripts/test.sh
enable_amp = True
empty_cache = True
mix_prob = 0.0
param_dicts = None
ddp_static_graph = True  # backward() runs several times per forward (per chunk)
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1),
]
train = dict(type="DefaultTrainer")

# model
model = dict(
    type="StreamingSegmentorV2",
    num_classes=49,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3-LaCT-Temporal",
        lact_head_dim=64,
        lact_base_lr=1e-2,
        lact_enc_stages=[1, 2, 3, 4],
        lact_dec_stages=[0, 1, 2, 3],
        train_with_ttt=True,
        max_batch_size_per_gpu=4,
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
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=1e-4,
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
        sample_unit="sequence",  # one full 300-frame sequence per sample, 2,221 samples
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment", "clip_idx", "frame_emb"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "clip_idx"),
                feat_keys=("coord",),
            ),
        ],
    ),
)
