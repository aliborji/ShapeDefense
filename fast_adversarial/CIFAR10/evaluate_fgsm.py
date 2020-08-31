import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from preact_resnet import * #PreActResNet18
from utils import (clamp, get_loaders,attack_pgd, evaluate_pgd, evaluate_standard)
from utils import detect_edge_batch
from config import *

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--net-type', default='rgb', type=str, choices=['rgb', 'edge', 'rgbedge'])
    parser.add_argument('--redetect-test', action='store_true', help='redetect edges')        
    parser.add_argument('--model', default='model.pth', type=str)


    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    _, test_loader = get_loaders(args.data_dir, args.batch_size)

   # model_test = PreActResNet18(args.net_type).cuda()
    model_test = define_model(args.net_type).cuda()# PreActResNet18(args.net_type)#.cuda()
    best_state_dict = torch.load(args.model)
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 10, args.net_type, args.redetect_test)
    test_loss, test_acc = evaluate_standard(test_loader, model_test, args.net_type)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
