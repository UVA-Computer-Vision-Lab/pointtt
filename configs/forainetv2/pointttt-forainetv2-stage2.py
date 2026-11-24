"""PoinTTT on FOR-instanceV2, stage 2: the stage-1 model plus 8 LaCT (TTT) layers, trained jointly.

Starts from model/model_best.pth of stage 1 (pointttt-forainetv2-stage1.py) and lists only what
differs from it: TTT layers on encoder stages 1-4 and decoder stages 0-3 with the fast weights
carried over all chunks of a plot (train_with_ttt=True), 200 epochs x 47 iterations, one learning
rate, a new seed. The 28 TTT keys that the stage-1 checkpoint lacks start from their seeded
initialisation (the loader logs them as missing keys).

    bash scripts/train.sh -d forainetv2 -c pointttt-forainetv2-stage2 -g 1 -n exp/forainetv2/stage2 \
        -w exp/forainetv2/stage1/model/model_best.pth

The released checkpoint pointttt_forainetv2.pth comes from the stage-2 run with this config, about
9.5 hours on one NVIDIA B200 including the 200 evaluations, peak GPU memory about 20 GB
(nvidia-smi). Evaluate a checkpoint with
pointttt-forainetv2-eval.py (scripts/test.sh).
"""

_base_ = ["./pointttt-forainetv2-stage1.py"]

weight = "exp/forainetv2/stage1/model/model_best.pth"  # stage 1; scripts/train.sh -w overrides it
seed = 50871921
epoch = 200  # data loop = epoch // eval_epoch = 1: one pass over the 47 plots per epoch
eval_epoch = 200
param_dicts = None
save_every_epoch = False  # keep epoch_10, epoch_20, ... (CheckpointSaver save_freq=10), model_best, model_last

model = dict(
    backbone=dict(
        lact_enc_stages=[1, 2, 3, 4],
        train_with_ttt=True,
    ),
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=1e-4,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
