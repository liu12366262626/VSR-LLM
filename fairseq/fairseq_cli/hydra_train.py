#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import debugpy

# # 设置debugpy监听特定端口
# debugpy.listen(('0.0.0.0', 5678))  # 使用0.0.0.0允许从任何IP连接
# print("⚠️ Debugger waiting for attachment at port 5678")
# debugpy.wait_for_client()  # 代码将在这里暂停，直到调试器连接


import logging
import os
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq_cli.train import main as pre_main
from fairseq import distributed_utils, metrics
from fairseq.dataclass.configs import FairseqConfig
from fairseq.utils import reset_logging

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, open_dict




logger = logging.getLogger("fairseq_cli.hydra_train")


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    print("Merged configuration:")
    print(OmegaConf.to_yaml(cfg))

    # 打印合并的配置源信息


    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        with open_dict(cfg):
            # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
            cfg.job_logging_cfg = OmegaConf.to_container(HydraConfig.get().job_logging, resolve=True)

    # 枚举里面所有${}这种变量并赋值
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    # 定死cfg里面的key
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, pre_main)
        else:
            distributed_utils.call_main(cfg, pre_main)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if best_val is None:
        best_val = float("inf")

    return best_val


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()


