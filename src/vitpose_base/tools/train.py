# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import re
import time
import warnings
from typing import Any, Dict, Optional

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, set_random_seed
from mmcv.utils import get_git_hash

from mmpose import __version__
from mmpose.apis import init_random_seed, single_gpu_test, train_model
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger, setup_multi_processes

import mmcv_custom  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=None,
        help='override log_config.interval to control training progress logs')
    parser.add_argument(
        '--status-file',
        default=None,
        help='write current training progress to this text file')
    parser.add_argument(
        '--status-interval',
        type=int,
        default=None,
        help='iterations between status-file updates; defaults to log interval')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    parser.add_argument(
        '--final-test',
        action='store_true',
        help='run a final evaluation on cfg.data.test after training finishes')
    parser.add_argument(
        '--final-test-checkpoint',
        default=None,
        help='checkpoint to evaluate for --final-test; defaults to best_* then latest.pth under work_dir')
    parser.add_argument(
        '--final-test-out',
        default=None,
        help='write final test metrics JSON to this path (default: <work_dir>/test_metrics.json)')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _epoch_num(path: str) -> int:
    m = re.search(r'epoch_(\d+)', osp.basename(path))
    return int(m.group(1)) if m else -1


def _pick_final_test_checkpoint(work_dir: str) -> Optional[str]:
    best_ckpts = [
        osp.join(work_dir, p)
        for p in mmcv.scandir(work_dir, suffix='.pth', recursive=False)
        if p.startswith('best_')
    ]
    if best_ckpts:
        return sorted(best_ckpts, key=_epoch_num)[-1]

    latest = osp.join(work_dir, 'latest.pth')
    if osp.isfile(latest):
        return latest

    epoch_ckpts = [
        osp.join(work_dir, p)
        for p in mmcv.scandir(work_dir, suffix='.pth', recursive=False)
        if p.startswith('epoch_')
    ]
    if epoch_ckpts:
        return sorted(epoch_ckpts, key=_epoch_num)[-1]

    return None


def _run_final_test(cfg: Config, checkpoint_path: str) -> Dict[str, Any]:
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    loader_cfg: Dict[str, Any] = dict(seed=cfg.get('seed'), drop_last=False, dist=False)
    test_loader_cfg: Dict[str, Any] = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {}),
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    model = build_posenet(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader)

    eval_config: Dict[str, Any] = cfg.get('evaluation', {}).copy()
    eval_config.update(dict(metric=['mAP']))
    return dataset.evaluate(outputs, cfg.work_dir, **eval_config)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.log_interval is not None:
        if 'log_config' not in cfg:
            cfg.log_config = dict(interval=args.log_interval, hooks=[dict(type='TextLoggerHook')])
        else:
            cfg.log_config['interval'] = args.log_interval

    if args.status_file is not None:
        status_interval = args.status_interval
        if status_interval is None:
            status_interval = args.log_interval
        if status_interval is None:
            status_interval = cfg.get('log_config', {}).get('interval', 50)
        cfg.status_monitor = dict(path=args.status_file, interval=max(1, int(status_interval)))

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            ' is deprecated because we only support '
            'single GPU mode in non-distributed training. '
            'Use  now.')

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            ' is deprecated, please use . '
            'Because we only support single GPU mode in '
            'non-distributed training. Use the first GPU '
            'in  now.')

    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta: Dict[str, Any] = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    try:
        config_text = cfg.pretty_text
    except TypeError:
        config_text = cfg.text
    logger.info(f'Config:\n{config_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=config_text,
        )

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    if args.final_test:
        if distributed:
            logger.warning('Skipping --final-test in distributed mode.')
            return

        checkpoint_path = args.final_test_checkpoint
        if checkpoint_path is None:
            checkpoint_path = _pick_final_test_checkpoint(cfg.work_dir)

        if checkpoint_path is None:
            logger.warning(f'--final-test requested but no checkpoint found under: {cfg.work_dir}')
            return

        metrics = _run_final_test(cfg, checkpoint_path)
        ap = metrics.get('AP')
        ap50 = metrics.get('AP50')
        logger.info(f'Final test checkpoint: {checkpoint_path}')
        logger.info(f'Final test metrics: AP(mAP50-95)={ap} AP50(mAP50)={ap50}')

        out_path = args.final_test_out or osp.join(cfg.work_dir, 'test_metrics.json')
        mmcv.dump(metrics, out_path)
        logger.info(f'Wrote final test metrics JSON: {out_path}')


if __name__ == '__main__':
    main()
