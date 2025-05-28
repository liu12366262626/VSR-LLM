#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
# 检查DEBUG_MODE环境变量
debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
print(f'debug_mode : {debug_mode}')
if debug_mode:
    import debugpy
    # 设置debugpy监听特定端口
    debugpy.listen(('127.0.0.1', 5679))  # 使用0.0.0.0允许从任何IP连接
    print("⚠️ Debugger waiting for attachment at port 5679")
    debugpy.wait_for_client()  # 代码将在这里暂停，直到调试器连接

import argparse
import logging
import math
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable
sys.path.append('/work/liuzehua/task/VSR/VSP-LLM/fairseq/fairseq')

# Setup root logger before importing any Fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch

from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)

from fairseq.data import iterators, data_utils
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.distributed as dist
from tqdm import tqdm
# 强制 tqdm 输出到标准错误流
tqdm(progress_bar, file=sys.stderr)



def initialize_distributed(cfg):
    """
    Initialize distributed training environment with added logging for debug.
    """
    logger.info("Initializing distributed environment...")

    # 使用环境变量来获取 `LOCAL_RANK`，`RANK` 和 `WORLD_SIZE`
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 本地进程 rank，指定当前 GPU ID
    global_rank = int(os.environ.get("RANK", 0))  # 全局进程 rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 总进程数

    # 将 local_rank 作为设备ID，指定要使用的GPU
    cfg.distributed_training.device_id = local_rank
    torch.cuda.set_device(local_rank)  # 设置当前进程的GPU设备

    # 如果分布式进程组没有初始化，则进行初始化
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',  # 使用 NCCL 后端进行 GPU 通信
            init_method='env://',  # 使用环境变量进行初始化
            world_size=world_size,  # 从环境变量中获取总进程数
            rank=global_rank  # 从环境变量中获取当前进程的 rank
        )
    
    logger.info(f"Distributed process initialized: rank {dist.get_rank()}, world_size {dist.get_world_size()}, local_rank {local_rank}")
    
# Hydra will automatically load the configuration from the specified YAML file
@hydra.main(config_path="/work/liuzehua/task/VSR/VSP-LLM/exp/model_v4/conf", config_name="vsp-llm-90h-single.yaml")
def main(cfg: FairseqConfig) -> None:
    """
    Main training function that reads configuration and starts distributed training.
    """
    # 如果配置中设置了分布式训练，初始化分布式环境
    if cfg.distributed_training.distributed_world_size > 1:
        initialize_distributed(cfg)
    # 添加用户自定义模块路径
    utils.import_user_module(cfg.common)
    
    # 重置训练指标
    metrics.reset()
    
    # 设置随机种子
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # 设置随机种子
    utils.set_torch_seed(cfg.common.seed)

    # 初始化任务
    task = tasks.setup_task(cfg.task)

    # 构建模型和损失函数
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    
    
    # 打印模型和任务信息
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )
    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    task.load_dataset(cfg.dataset.valid_subset, combine=False, epoch=1)
    
    quantizer = None


    # 构建训练器,模型过大时，可以用多个GPU加载一个模型
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    # logger.info(
    #     "max tokens per device = {} and max sentences per device = {}".format(
    #         # cfg.dataset.max_tokens,
    #         cfg.dataset.batch_size,
    #     )
    # )
    
    # 加载最新的检查点（如果存在）
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    
    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break
    
        # 训练一个 epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break
    
        # 更新学习率
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
    
        # 获取下一个 epoch 的迭代器
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            load_dataset=task.has_sharded_data("train"),
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))
    
    # 等待所有异步检查点保存完成
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # 如果当前没有验证损失，则不停止
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # 初始化数据迭代器
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=True,
    )
    
    # 设置梯度累积步数
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    
    
    progress = progress_bar.progress_bar(
        itr,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm"),
    )

    progress.update_config(_flatten_config(cfg))
    
    trainer.begin_epoch(epoch_itr.epoch)
    
    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)
    
        if log_output is not None:  # not OOM, overflow, ...
            # 记录中间日志
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)
    
                # 重置中间指标
                metrics.reset_meters("train_inner")
    
        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
    
        if should_stop:
            break
    
    # 记录 epoch 结束的指标
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)
    
    # 重置 epoch 级别的指标
    metrics.reset_meters("train")
    return valid_losses, should_stop

#将配置文件转换为普通的字典
def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # 移除任何遗留的 Namespace 并替换为单一的 "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf
    
    # 停止条件
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )
    
    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )
    
    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # mid-epoch 验证
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation and num_updates >= cfg.dataset.validate_after_updates
    
    # 验证
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
    
    should_stop |= should_stop_early(cfg, valid_losses[0])
    
    # 保存检查点
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )
    
    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""
    
    if cfg.dataset.fixed_validation_seed is not None:
        # 为每次验证设置固定种子
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)
    
    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info(f'begin validation on "{subset}" subset')
    
        # 初始化数据迭代器
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # 使用固定的验证集
        )

        progress = progress_bar.progress_bar(
            itr,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm"),
        )
    
        # 创建一个新的根指标聚合器，以便验证指标不会污染其他聚合器（例如训练指标）
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break
                trainer.valid_step(sample)
    
        # 记录验证指标
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
    
        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)
    
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
    
        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = f"best_{cfg.checkpoint.best_checkpoint_metric}"
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


    


if __name__ == "__main__":
    main()
