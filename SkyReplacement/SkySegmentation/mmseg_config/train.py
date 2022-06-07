# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes

import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config',help='test config file path. (EX) pspnet.py')
    parser.add_argument('--work-dir',default='./work_dirs' ,help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from, If you are using some checkpoint as pretrain, you should use load_from.')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from, When training is interrupted somehow, resume_from should be used to resume training.')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    parser.add_argument('--debug', dest = 'debug',  action='store_true', help = 'Debug mode which do not run wandb and epoch 2 ')
    parser.set_defaults(debug=False)

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--tags', nargs='+', default=[],
        help ='record your experiment speical keywords into tags list'
        '--tags batch_size=16 swin_cascasde'
        "dont use white space in specific tag") 

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.'
        ' If you want to auto resume with latest checkpoint, use this option'
        'Fine latest checkpoint at cfg.work_dir'
        'https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/train.py#L249')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_root = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/configs/_base_'

    cfg = Config.fromfile(os.path.join(config_root,args.config))
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    

    # create work_dir -> ./workdir if you already set, comment out this line
    work_dir = utils.increment_path(os.path.join(args.work_dir,'exp'))
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    cfg.work_dir = work_dir

    # resume_from or load_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from


    # set gpu_id, we only use one gpu. if you want more gpu and setting, reference mmsegmentation/tools/train.py
    cfg.gpu_ids = [0]

    cfg.auto_resume = args.auto_resume

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set random seeds
    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=True)
    cfg.seed = seed


    # wandb
    cfg.log_config['hooks'][1]['init_kwargs']['tags'] = args.tags #args를 그냥 보내서 바뀐 것들은 이걸로 표현해도 나쁘진 않을 듯.
    cfg.log_config['hooks'][1]['init_kwargs']['name'] = work_dir.split('/')[-1]

    cfg.log_config['hooks'][1]['init_kwargs']['config'] = cfg   

    if args.debug : # args.wandb is False -> wandb don't work maybe default = True
        cfg.log_config['hooks']=[dict(type='TextLoggerHook')]
        cfg.runner['max_epochs']=2

    meta = dict()
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]

    train_segmentor(
        model,
        datasets,
        cfg,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta = meta)


if __name__ == '__main__':
    # import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
