import sys
import argparse
import pprint
import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm

sys.path.append("./")
import lib
from lib.utils.utils import create_logger, speed_test
from lib.models.ddrnet_23_slim import get_seg_model
from lib.datasets import *
from lib.config import config
from lib.config import update_config

root = os.path.abspath(os.path.join(os.getcwd()))


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    cfg_path = os.path.join(root, "experiments/cityscapes/ddrnet23_slim.yaml")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=cfg_path,
                        type=str)
    parser.add_argument('--seed', type=int, default=34)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args



class MscEvalV0(object):

    def __init__(self, scale=1, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label, _, name, *border_padding) in diter:
            N, _, H, W = label.unsqueeze(1).shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H * self.scale), int(W * self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[1]

            logits = F.interpolate(logits, size=size,
                                   mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()




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

    ## dataset
    batchsize = 8
    n_workers = 2

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dsval = eval('lib.datasets.' + config.DATASET.DATASET)(
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
    dl = DataLoader(dsval,
                    batch_size=batchsize,
                    shuffle=False,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=False)

    model = get_seg_model(config)
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')


    model_state_file = os.path.join(root, model_state_file)
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

    model.cuda()
    model.eval()

    with torch.no_grad():
        single_scale = MscEvalV0()
        mIOU = single_scale(model, dl, 19)

    logger.info('batch-size is:{}, mIOU is: {:4.4f}\n'.format(batchsize, mIOU))

if __name__ == '__main__':
    main()


