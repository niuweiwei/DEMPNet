# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


sys.path.append("./")
import lib
from lib.config import config
from lib.config import update_config
from lib.core.function import testval, test,validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, speed_test
from lib.models import *
from lib.datasets import *

root = os.path.abspath(os.path.join(os.getcwd()))

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
<<<<<<< HEAD
    cfg_path = os.path.join(root, "experiments/cityscapes/ddrnet23_slim.yaml")
=======
    cfg_path = os.path.join(root, "experiments/camvid/ddrnet23_slim.yaml")
>>>>>>> e4abc71a3d00cde32d34f9f3749ddaac85052449
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=cfg_path,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('lib.models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('lib.models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')      
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')
<<<<<<< HEAD

    model_state_file = os.path.join(root,model_state_file)
=======
>>>>>>> e4abc71a3d00cde32d34f9f3749ddaac85052449
    print(model_state_file)
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
<<<<<<< HEAD
    batch_size = 1
=======
>>>>>>> e4abc71a3d00cde32d34f9f3749ddaac85052449
    test_dataset = eval('lib.datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
<<<<<<< HEAD
        batch_size=batch_size,
=======
        batch_size=1,
>>>>>>> e4abc71a3d00cde32d34f9f3749ddaac85052449
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
                                                        test_dataset, 
                                                        testloader, 
                                                        model,
<<<<<<< HEAD
                                                        sv_dir= final_output_dir,
                                                        sv_pred=False)

    msg = 'BatchSize: {}, MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(batch_size, mean_IoU,
=======
                                                        sv_pred=False)

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
>>>>>>> e4abc71a3d00cde32d34f9f3749ddaac85052449
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
