"""
Launcher

modified from detectron2(https://github.com/facebookresearch/detectron2)
Modified for PoinTTT: single machine only (one process per GPU, rendezvous on a free local port).

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pointcept.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=60)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(main_func, num_gpus, cfg=(), timeout=DEFAULT_TIMEOUT):
    """
    Launch single- or multi-gpu training on this machine.
    With more than one GPU it spawns one process per GPU.
    Args:
        main_func: a function that will be called by `main_func(*cfg)`
        num_gpus (int): number of GPUs (processes)
        cfg (tuple): arguments passed to main_func
        timeout (timedelta): timeout of the distributed workers
    """
    if num_gpus > 1:
        dist_url = f"tcp://127.0.0.1:{_find_free_port()}"
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus,
            args=(main_func, num_gpus, dist_url, cfg, timeout),
            daemon=False,
        )
    else:
        main_func(*cfg)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    dist_url,
    cfg,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group (all ranks: one machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    comm._LOCAL_PROCESS_GROUP = dist.new_group(list(range(world_size)))

    assert world_size <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*cfg)
