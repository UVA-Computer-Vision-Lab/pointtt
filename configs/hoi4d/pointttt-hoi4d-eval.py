"""PoinTTT on HOI4D: evaluation of the released checkpoint on the 892 test sequences.

Causal online inference over 3-frame chunks, predictions mapped to all
300 x 8192 raw points of every sequence by nearest neighbour. The reported number, pptr_mIoU, is
computed from the log by scripts/pptr_miou.py.

    bash scripts/test.sh -d hoi4d [-g 4] [-w path/to/pointttt_hoi4d.pth]

The defaults below are the reference setting: 4 GPUs x 8 sequences per GPU (batch_size = 32),
2 loader workers per GPU (num_worker = 8). scripts/test.sh sets batch_size,
batch_size_val and num_worker for other GPU counts.
"""

# runtime
weight = "checkpoints/pointttt_hoi4d.pth"
resume = False
evaluate = True
is_train = False
seed = 32069811
save_path = "exp/hoi4d_eval"
num_worker = 8  # loader workers in total over all GPUs
batch_size = 32  # sequences in total over all GPUs, at most 8 per GPU (max_batch_size_per_gpu)
batch_size_val = 32
batch_size_test = None
epoch = 0
eval_epoch = 1
sync_bn = False
enable_amp = True
empty_cache_per_epoch = False
find_unused_parameters = False
ddp_static_graph = True
hooks = [
    dict(type="CheckpointLoader", strict=True),
    dict(type="SemSegEvaluator"),
]
train = dict(type="DefaultTrainer")

# model
model = dict(
    type="StreamingSegmentorV2",
    num_classes=49,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3-LaCT-Temporal",
        chunk_size=10240,
        lact_head_dim=64,
        lact_base_lr=1e-2,
        lact_use_muon=True,
        lact_enc_stages=[1, 2, 3, 4],
        lact_dec_stages=[0, 1, 2, 3],
        freeze_backbone=False,
        train_with_ttt=True,
        max_batch_size_per_gpu=8,
        ttt_online_mode=True,
        ttt_online_view_avg=True,
        ttt_online_extra_views=2,
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
        pdnorm_conditions=("HOI4D",),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),  # only for the logged val loss
    ],
)

# data
data_root = "data/hoi4d"  # test1_right.h5 + test2_right.h5 (see README)
data = dict(
    num_classes=49,
    ignore_index=-1,
    names=[str(i) for i in range(49)],
    train=dict(),  # not used for evaluation (tools/train.py expects the key)
    val=dict(
        type="HOI4DDataset",
        sample_unit="sequence",
        split="val",  # test1_right.h5 + test2_right.h5
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_segment"}),
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
                keys=("coord", "grid_coord", "segment", "clip_idx", "origin_coord", "origin_segment"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
)
