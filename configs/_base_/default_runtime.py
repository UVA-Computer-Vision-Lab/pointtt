"""Runtime defaults of the training configs.

Pointcept's configs/_base_/default_runtime.py, reduced to the entries that not every training
config sets itself, plus is_train = True (tools/test.py refuses such configs) and
grad_accum_steps = 1.
"""

weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
batch_size_val = None  # auto adapt to bs 1 for each gpu
is_train = True
grad_accum_steps = 1
