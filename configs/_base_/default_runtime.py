"""Runtime defaults of the training configs.

Pointcept's configs/_base_/default_runtime.py, with the hooks of this repository (no
PreciseEvaluator) and without the `test` entry (this repository has no tester; evaluation runs
through tools/test.py with an evaluation config), plus is_train = True (tools/test.py refuses such
configs) and grad_accum_steps = 1.
"""

weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch
clip_grad = None  # disable with None, enable with a float

sync_bn = False
enable_amp = False
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

# Trainer
train = dict(type="DefaultTrainer")
is_train = True
grad_accum_steps = 1
