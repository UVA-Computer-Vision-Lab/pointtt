"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)
Modified for PoinTTT: single-machine command line (--config-file, --num-gpus, --options) and
the setup steps the configs of this repository use.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
from torch.nn.parallel import DistributedDataParallel


import pointcept.utils.comm as comm
from pointcept.utils.env import set_seed
from pointcept.utils.config import Config, DictAction


def create_ddp_model(model, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    kwargs["device_ids"] = [comm.get_local_rank()]
    kwargs["output_device"] = [comm.get_local_rank()]
    return DistributedDataParallel(model, **kwargs)


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def default_config_parser(file_path, options):
    cfg = Config.fromfile(file_path)

    if options is not None:
        cfg.merge_from_dict(options)

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = cfg.seed * cfg.num_worker_per_gpu + rank
    set_seed(seed)
    return cfg
