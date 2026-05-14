# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, Hook,
                         OptimizerHook, get_dist_info)
from mmcv.utils import digit_version

from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook


class TrainingStatusHook(Hook):
    def __init__(self, path, interval=50, train_iters_per_epoch=None):
        self.path = path
        self.interval = max(1, int(interval))
        self.train_iters_per_epoch = train_iters_per_epoch
        self.started_at = None

    def _write(self, runner, phase, message=None):
        now = time.time()
        if self.started_at is None:
            self.started_at = now

        epoch = runner.epoch + 1
        inner_iter = runner.inner_iter + 1
        max_epochs = runner.max_epochs
        total_iters = None
        progress_pct = None
        eta_seconds = None
        if self.train_iters_per_epoch:
            total_iters = self.train_iters_per_epoch * max_epochs
            completed_iters = min(runner.iter + 1, total_iters)
            progress_pct = (completed_iters / total_iters) * 100.0
            elapsed = max(0.0, now - self.started_at)
            if completed_iters > 0:
                eta_seconds = max(0.0, (elapsed / completed_iters) * (total_iters - completed_iters))

        lines = [
            f"updated_at={time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}",
            f"phase={phase}",
            f"epoch={epoch}",
            f"max_epochs={max_epochs}",
            f"iter_in_epoch={inner_iter}",
            f"iters_per_epoch={self.train_iters_per_epoch}",
            f"global_iter={runner.iter + 1}",
            f"total_iters={total_iters}",
            f"progress_pct={progress_pct:.2f}" if progress_pct is not None else "progress_pct=",
            f"eta_seconds={int(eta_seconds)}" if eta_seconds is not None else "eta_seconds=",
            f"work_dir={runner.work_dir}",
        ]
        if message:
            lines.append(f"message={message}")

        abs_path = os.path.abspath(self.path)
        status_dir = os.path.dirname(abs_path)
        os.makedirs(status_dir, exist_ok=True)
        tmp_path = f"{abs_path}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as status_file:
            status_file.write('\n'.join(lines) + '\n')
        os.replace(tmp_path, abs_path)

    def before_run(self, runner):
        self.started_at = time.time()
        self._write(runner, 'starting')

    def before_train_epoch(self, runner):
        self._write(runner, 'train')

    def after_train_iter(self, runner):
        if (runner.inner_iter + 1) % self.interval == 0:
            self._write(runner, 'train')

    def after_train_epoch(self, runner):
        self._write(runner, 'train_epoch_done')

    def after_run(self, runner):
        self._write(runner, 'finished')


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    train_dataloader = data_loaders[0]
    train_dataset = dataset[0]
    train_iters_per_epoch = len(train_dataloader)
    total_train_iters = train_iters_per_epoch * cfg.total_epochs

    # determine whether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if digit_version(mmcv.__version__) >= digit_version(
                '1.4.4') or torch.cuda.is_available():
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        else:
            warnings.warn(
                'We recommend to use MMCV >= 1.4.4 for CPU training. '
                'See https://github.com/open-mmlab/mmpose/pull/1157 for '
                'details.')

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    status_monitor = cfg.get('status_monitor', None)
    if status_monitor:
        runner.register_hook(
            TrainingStatusHook(
                path=status_monitor['path'],
                interval=status_monitor.get('interval', 50),
                train_iters_per_epoch=train_iters_per_epoch),
            priority='VERY_LOW')
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    else:
        val_dataset = None
        val_dataloader = None

    checkpoint_interval = None
    if cfg.get('checkpoint_config', None) is not None:
        checkpoint_interval = cfg.checkpoint_config.get('interval')
    eval_interval = None
    if validate and cfg.get('evaluation', None) is not None:
        eval_interval = cfg.evaluation.get('interval')

    logger.info(
        'Training progress summary: '
        f'train_samples={len(train_dataset)}, '
        f'batch_size={train_loader_cfg.get("samples_per_gpu")}, '
        f'workers={train_loader_cfg.get("workers_per_gpu")}, '
        f'iters_per_epoch={train_iters_per_epoch}, '
        f'total_epochs={cfg.total_epochs}, '
        f'total_train_iters={total_train_iters}')
    if validate and val_dataset is not None and val_dataloader is not None:
        logger.info(
            'Validation progress summary: '
            f'val_samples={len(val_dataset)}, '
            f'val_batch_size={dataloader_setting.get("samples_per_gpu")}, '
            f'val_iters={len(val_dataloader)}, '
            f'eval_interval={eval_interval}')
    logger.info(
        'Checkpoint summary: '
        f'work_dir={cfg.work_dir}, '
        f'checkpoint_interval={checkpoint_interval}, '
        f'resume_from={cfg.get("resume_from", None)}, '
        f'load_from={cfg.get("load_from", None)}')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
