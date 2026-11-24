"""
Evaluation entry point: runs an evaluation config (`is_train = False`) through the same
trainer as tools/train.py, so the numbers are those of tools/train.py with that config.

    python tools/test.py --config-file configs/forainetv2/pointttt-forainetv2-eval.py \
        --num-gpus 1 --options weight=checkpoints/pointttt_forainetv2.pth save_path=exp/forainetv2_eval

scripts/test.sh wraps it with the data and GPU checks of each dataset.
"""

import sys

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.utils.config import Config


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    # check before default_config_parser creates the output directory
    probe = Config.fromfile(args.config_file)
    if args.options is not None:
        probe.merge_from_dict(args.options)
    if probe.get("is_train", True):
        sys.exit(
            f"{args.config_file} is a training config (is_train = True); evaluate with an "
            "evaluation config, e.g. configs/<dataset>/pointttt-<dataset>-eval.py"
        )
    if not probe.get("weight"):
        sys.exit("no checkpoint given: pass --options weight=/path/to/checkpoint.pth")
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
