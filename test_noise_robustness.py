# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
from model import *

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--net_type', '-n', type=str,
                    choices=['rgb', 'rgbedge', 'edge'])

# Architecture
parser.add_argument('--model_name', '-m', type=str)
                    #choices=['tinyImgnet.pth'])
# Acceleration
# parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////

# if args.model_name == 'tinyImgnet':
#     net = resnext_50_32x4d
#     net.load_state_dict(torch.load(''))
#     args.test_bs = 64


data_dir = 'tiny-imagenet-200'
inp_size = 64
n_classes = 200


net_type = args.net_type.lower()

net, _, _, _ = build_model_resNet(net_type, data_dir, inp_size, n_classes)
net.load_state_dict(torch.load(args.model_name))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)




args.test_bs = 64

args.prefetch = 4

# for p in net.parameters():
#     p.volatile = True

# if args.ngpu > 1:
#     net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# if args.ngpu > 0:
#     net.cuda()

torch.manual_seed(1)
np.random.seed(1)
# if args.ngpu > 0:
#     torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

print('Model Loaded')

# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
#     root="./tiny-imagenet-200/testset/images/",
#     # transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
#     transform=trn.Compose([trn.Resize(64), trn.ToTensor(), trn.Normalize(mean, std)])),    
#     batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


# correct = 0
# for batch_idx, (data, target) in enumerate(clean_loader):
#     data = V(data.cuda(), volatile=True)
#
#     output = net(data)
#
#     pred = output.data.max(1)[1]
#     correct += pred.eq(target.cuda()).sum()
#
# clean_error = 1 - correct / len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []

    for severity in range(1, 6):
        distorted_dataset = dset.ImageFolder(
            # root='/share/data/vision-greg/DistortedImageNet/JPEG/' + distortion_name + '/' + str(severity),
            root='./Tiny-ImageNet-C/' + distortion_name + '/' + str(severity),                        
            # transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))
            transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)]))            

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            data, target = data.to(device), target.to(device)
            # data = V(data.cuda(), volatile=True)

            output = net(data)

            pred = output.data.max(1)[1]
            # correct += pred.eq(target.cuda()).sum()
            correct += pred.eq(target).sum()
 

        errs.append(1 - 1.*correct / len(distorted_dataset))

    print('\n=Average', tuple(errs))
    return np.mean(errs)


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]

error_rates = []
for distortion_name in distortions:
    print(distortion_name)
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))


print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))

