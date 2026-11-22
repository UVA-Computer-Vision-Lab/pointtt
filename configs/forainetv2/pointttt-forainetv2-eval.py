"""PoinTTT on FOR-instanceV2: evaluation of the released checkpoint on the 28 test plots.

Whole plots, 0.05 m voxels, chunk-wise inference (102,400 voxels per chunk), predictions mapped
back to every raw point by nearest neighbour, IoU accumulated over all plots, no test-time
augmentation, 1 GPU.

    bash scripts/eval_forest.sh [path/to/pointttt_forainetv2.pth]
"""

# runtime
weight = "checkpoints/pointttt_forainetv2.pth"
resume = False
evaluate = True
is_train = False
seed = 50871921
save_path = "exp/forainetv2_eval"
num_worker = 4
batch_size = 1
batch_size_val = None
batch_size_test = None
epoch = 0
eval_epoch = 1
sync_bn = False
enable_amp = False
empty_cache_per_epoch = False
find_unused_parameters = False
hooks = [
    dict(type="CheckpointLoader", strict=True),
    dict(type="SemSegEvaluator"),
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
        lact_enc_stages=[1, 2, 3, 4],
        lact_dec_stages=[0, 1, 2, 3],
        freeze_backbone=False,
        train_with_ttt=True,
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
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),  # only for the logged val loss
    ],
)

# data
data_root = "data/forainetv2"  # test_data/*.ply + meta_data/test_list.txt (scripts/prepare_forest.sh)
data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["ground", "wood", "leaf"],
    train=dict(),  # not used for evaluation (tools/train.py expects the key)
    val=dict(
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
