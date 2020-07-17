from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms


import sys
sys.path.insert(0,'..')

###############  IMPORTANT: Specify your edge detector here #######################################################
from edge_detector import *
edge_detect = compute_energy_matrix
###################################################################################################################

# from config import *
from dataset import MyDataset, Dataset_MNIST, Dataset_FashionMNIST, DogsDataset, folderDB, Dataset_CIFAR10
from model import build_model, build_model_mnist, build_model_dogs
from utils import detect_edge_batch, is_image_file, load_img, save_img





# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)



# _, dataloader_dict, _, _ = build_model_mnist(net_type='gray')
_, dataloader_dict, _, _ = build_model_dogs('rgbedge', '../dog-breed-identification/', 64)



# training_data_loader = dataloader_dict['train']
testing_data_loader = dataloader_dict['val']




count = 0
for idx, batch in enumerate(testing_data_loader):
    input = detect_edge_batch(batch[0]) # detect edges

    input = input.to(device)

    for i in range(input.shape[0]):
        # input = img.unsqueeze(0).to(device)
        # import pdb; pdb.set_trace()

        out = net_g(input[i,3][None,None]) #  get the last channel which is egde
        out_img = out.detach().squeeze(0).cpu()
        # import pdb; pdb.set_trace()
        # print(out_img.shape)

        out_img = out_img.permute(1,2,0)
        out_img = (out_img - out_img.min())/ (out_img.max() - out_img.min())
        out_img = out_img*255

        if not os.path.exists(os.path.join("result", opt.dataset)):
            os.makedirs(os.path.join("result", opt.dataset))
        save_img(out_img, "result/{}/{}".format(opt.dataset, str(idx)+'.jpg'))


        orig_edge = input[i,3]
        # orig_edge = (orig - orig.min())/ (orig.max() - orig.min())
        orig_edge = orig_edge*255

        save_img(orig_edge, "result/{}/{}".format(opt.dataset, str(idx)+'_orig.jpg'))


        orig_rgb = input[i,:3].permute(1,2,0)
        orig_rgb = (orig_rgb - orig_rgb.min())/ (orig_rgb.max() - orig_rgb.min())
        orig_rgb = orig_rgb*255

        save_img(orig_rgb, "result/{}/{}".format(opt.dataset, str(idx)+'_origRgb.jpg'))

        count +=1 