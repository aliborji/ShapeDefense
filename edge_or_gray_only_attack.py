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
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument('--attack', type=str, default='FGSM', help='attack type FGSM or PGD')
parser.add_argument('--net_type', type=str, default='rgb', help='edge, rgb, grayedge, rgbedge')
parser.add_argument('--data_dir', type=str, default='MNIST', help='data directory')
parser.add_argument('--model', type=str, default='mnist', help='which model implemented in model.py')
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')
parser.add_argument('--sigma', type=int, default=8, help='size of the input image')
parser.add_argument('--load_model', type=str, default='', help='path to the trained model')

opt = parser.parse_args()
print(opt)


# eg\
# python edge_only_attack.py --net_type edge --model MNIST  --sigma 32 --data_dir MNIST --load_model ResMNIST/FGSM/mnist_grayedge_32_robust_32.pth 


batch_size = opt.batch_size # 100
attack_type = opt.attack #'FGSM'
net_type = opt.net_type #'grayedge'
data_dir = opt.data_dir #'MNIST'
n_classes = opt.classes # 10
inp_size = opt.inp_size# 28
eps_t = opt.sigma
which_model = opt.model #'mnist'



if not os.path.exists(f'./Res{data_dir}'):
    os.mkdir(f'./Res{data_dir}')

if not os.path.exists(f'./Res{data_dir}/{attack_type}'):
    os.mkdir(f'././Res{data_dir}/{attack_type}')


fo = open(f'./Res{data_dir}/{attack_type}/results_EdgeOnly_{net_type}.txt', 'a+')

# --------------------------------------------------------------------------------------------------------------------------------------------

# # Test the clean model on clean and attacks
# acc, images = test_model_clean(net, dataloader_dict)
# print('Accuracy of original model on clean images: %f ' % acc)
# fo.write('Accuracy of original model on clean images: %f \n' % acc)




print(f'eps_t={eps_t}')
fo.write(f'eps_t={eps_t} \n')
fo.write(f'{opt.load_model} \n')

epsilons = [eps_t/255]


# Train a model first
# load the original model first
net, dataloader_dict, _, _ = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
load_model(net, opt.load_model)
net.to(device)


acc_attack, _ = test_model_edge_only(net, dataloader_dict, epsilons, attack_type)
print('Accuracy of the when masking attacked BG pixels: %f' % acc_attack[0])
fo.write('Accuracy of the when masking attacked BG pixels: %f \n' % acc_attack[0])



# acc_attack, _ = test_model_gray_only(net, dataloader_dict, epsilons, attack_type)
# print('Accuracy of the when masking attacked BG pixels: %f' % acc_attack[0])
# fo.write('Accuracy of the when masking attacked BG pixels: %f \n' % acc_attack[0])



fo.close()

