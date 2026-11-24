"""PoinTTT on FOR-instanceV2, stage 1: chunk-wise PTv3 trained from scratch.

The backbone is PT-v3-LaCT-Dec with its TTT layers switched off (train_with_ttt=False): whole
plots, 0.05 m voxels, 102,400 voxels per chunk, one plot per iteration, 1 GPU, fp32. The model is
evaluated after every trainer epoch; model/model_best.pth initialises stage 2
(pointttt-forainetv2-stage2.py).

    bash scripts/train.sh -d forainetv2 -c pointttt-forainetv2-stage1 -g 1 -n exp/forainetv2/stage1

Schedule: 40 trainer epochs x 470 iterations (10 passes over the 47 training plots each, 400 in
total). The stage-1 run behind the released checkpoint used exactly this config (seed included),
about 8.5 hours on one NVIDIA B200 including the 40 evaluations, peak GPU memory about 16 GB
(nvidia-smi). Every epoch is saved
(model/epoch_N.pth, about 0.55 GB each); stage 2 needs only model_best.pth, so
-o "save_every_epoch=False" (keep every 10th) saves disk without changing the training.

lact_enc_stages / lact_dec_stages create the TTT layers of stages 3-4 and 0-3 although they are not
used here: they are part of the seeded initialisation and of the checkpoint that stage 2 loads.
"""

_base_ = ["../_base_/default_runtime.py"]

# runtime
seed = 13235636
num_worker = 4
batch_size = 1  # one whole plot per iteration
epoch = 400  # passes over the training plots; data loop = epoch // eval_epoch = 10
eval_epoch = 40  # trainer epochs, each followed by an evaluation on the test split
clip_grad = 1.0
enable_amp = False
empty_cache = True
mix_prob = 0.0
param_dicts = [dict(keyword="block", lr=6e-5)]
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=10),
]
train = dict(type="DefaultTrainer")

# model
model = dict(
    type="StreamingSegmentorV2",
    num_classes=3,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3-LaCT-Dec",
        chunk_size=102400,
        lact_head_dim=64,
        lact_base_lr=1e-2,
        lact_use_muon=True,
        lact_enc_stages=[3, 4],
        lact_dec_stages=[0, 1, 2, 3],
        freeze_backbone=False,
        train_with_ttt=False,
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
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# optimizer
optimizer = dict(type="AdamW", lr=6e-4, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[6e-4, 6e-5],  # [all other parameters, transformer blocks]
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# data
data_root = "data/forainetv2"  # train_val_data/, test_data/, meta_data/ (scripts/prepare_forest.sh --train)
data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["ground", "wood", "leaf"],
    train=dict(
        type="ForAINetV2Dataset",
        split="train",  # the 47 plots of meta_data/train_list.txt
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
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment", "instance"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord",)),
        ],
        test_mode=False,
    ),
    val=dict(  # the test split, with the pipeline of the evaluation config
        type="ForAINetV2Dataset",
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment", "instance"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_coord", "origin_segment"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
)
