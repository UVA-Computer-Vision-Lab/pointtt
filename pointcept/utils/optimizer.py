"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


def build_optimizer(cfg, model, param_dicts=None):
    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            cfg.params.append(dict(names=[], params=[], lr=param_dicts[i].lr))

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)
