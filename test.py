from __future__ import print_function
import argparse

from lib import *
from config import *
from model import model_dispatcher
from utils import * 
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




###############  Substitue / black box attack ! #######################################################




###############  IMPORTANT: Specify your edge detector in config.py #######################################################

# Training settings
parser = argparse.ArgumentParser(description='shape defence adv training')
parser.add_argument('--attack', type=str, default='FGSM', help='attack type FGSM or PGD')
parser.add_argument('--net_type', type=str, default='rgb', help='edge, rgb, grayedge, rgbedge')
parser.add_argument('--data_dir', type=str, default='MNIST', help='data directory')
parser.add_argument('--model', type=str, default='mnist', help='which model implemented in model.py')
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')
parser.add_argument('--sigma', type=int, default=8, help='size of the input image')
parser.add_argument('--load_model', type=str, default='', help='path to the trained model')
parser.add_argument('--redetect_edge', action='store_true', default=False)

opt = parser.parse_args()
print(opt)


# eg\
# python test.py --net_type gray --model MNIST  --sigma 8 --data_dir MNIST --classes 10 --inp_size 28 --load_model mnist_gray.pth

attack_type = opt.attack #'FGSM'
net_type = opt.net_type #'grayedge'
data_dir = opt.data_dir #'MNIST'
which_model = opt.model #'mnist'
n_classes = opt.classes # 10
inp_size = opt.inp_size# 28
eps_t = opt.sigma
redetect_edge = opt.redetect_edge



print(f'eps_t={eps_t}')

epsilons = [eps_t/255]


# Train a model first
# load the original model first
net, dataloader_dict, _, _ = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
load_model(net, opt.load_model)
net.to(device)



# import pdb; pdb.set_trace()
# acc_attack, _ = test_model_clean(net, dataloader_dict)
# print('Accuracy of the original model on clean images: %f' % acc_attack)

acc_attack, _ = test_model_blackout_img(net, dataloader_dict)
print('Accuracy of the orig robust model on black out images: %f' % acc_attack[0])



# acc_attack, _ = test_model_image_edge_attack(net, dataloader_dict, epsilons, attack_type, net_type)
# print('Accuracy of the orig robust model on adversarial images: %f' % acc_attack[0])
