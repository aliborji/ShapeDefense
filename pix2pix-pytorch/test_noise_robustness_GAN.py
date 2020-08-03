# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')


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
from utils import detect_edge_batch, save_img
from dataset import folderDB
from make_imagenet_64_c import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--net_type', '-n', type=str,
                    choices=['rgb', 'rgbedge', 'edge'])

parser.add_argument('--data_dir', type=str)
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')

# Architecture
parser.add_argument('--model_name', '-m', type=str) # classifier trained on generated images
parser.add_argument('--gan_model', type=str, default='', help='path to the gan trained model')

# Acceleration
# parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
args = parser.parse_args()
print(args)


# python test_noise_robustness_GAN.py --n rgb -m ./checkpoint/Icons/Icons_rob.pth --data_dir ../Icons-50 --classes 50 --inp_size 64 --gan_model checkpoint/Icons/netG_model_epoch_95.pth



# /////////////// Model Setup ///////////////

# if args.model_name == 'tinyImgnet':
#     net = resnext_50_32x4d
#     net.load_state_dict(torch.load(''))
#     args.test_bs = 64


data_dir = args.data_dir # 'tiny-imagenet-200'
# inp_size = 64
# n_classes = 50 #200


net_type = args.net_type.lower()

net, _, _, _ = build_model_resNet(net_type, args.data_dir, args.inp_size, args.classes)
net.load_state_dict(torch.load(args.model_name,map_location=torch.device('cpu')))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)



net_g = torch.load(args.gan_model).to(device)



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

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

mean = [0.3403, 0.3121, 0.3214]
std = [0.2724, 0.2608, 0.2669]


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area




def show_performance(distortion_name):
    errs = []


    transform=trn.Compose([ trn.ToTensor(), trn.Normalize(mean, std)])
    # transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)])

    for severity in range(1,6):
        distorted_dataset = folderDB(
            root_dir='.', train=False,  transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor()]), net_type=net_type, base_folder=data_dir)

        if net_type == 'edge':    
            distorted_dataset = folderDB(
                root_dir='.', train=False,  transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor()]), net_type='rgbedge', base_folder=data_dir)



        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=2)



        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            data, target = data.to(device), target.to(device)

            for idx,d in enumerate(data):
                # import pdb; pdb.set_trace()                
                exr = distortion_name + '(d.permute(1,2,0)*255,severity)'
                aa = torch.Tensor(eval(exr))
                data[idx] = transform((aa/255).numpy())

            edge_maps = torch.zeros((data.shape[0],1,data.shape[2],data.shape[2]))
            data = torch.cat((data, edge_maps),dim=1)#[None]

            data = detect_edge_batch(data.detach())
            edge_maps = data[:,-1].unsqueeze(1)

            data = net_g(edge_maps) # pass the images through the GAN



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
    'defocus_blur', 'glass_blur', 
    'motion_blur', 
    'zoom_blur',
    'snow', 
    'frost', 
    'fog', 
    'brightness',
    'contrast', 
    'elastic_transform', 
    'pixelate', 
    'jpeg_compression',
    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]



error_rates = []
for distortion_name in distortions:
    print(distortion_name)
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))


print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))

