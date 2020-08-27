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
parser.add_argument('--epochs', type=int, default=20, help='num epochs')
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument('--attack', type=str, default='FGSM', help='attack type FGSM or PGD')
parser.add_argument('--net_type', type=str, default='rgb', help='edge, rgb, grayedge, rgbedge')
parser.add_argument('--net_type_substitute', type=str, default='rgb', help='edge, rgb, grayedge, rgbedge')
parser.add_argument('--data_dir', type=str, default='MNIST', help='data directory')
parser.add_argument('--model', type=str, default='mnist', help='which model implemented in model.py')
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')
parser.add_argument('--sigma', type=int, default=8, help='size of the input image')
parser.add_argument('--load_model', type=str, default='', help='path to the trained model')
parser.add_argument('--redetect_edge', action='store_true', default=True)

opt = parser.parse_args()
print(opt)


# eg\
# python BPDA.py --net_type grayedge --model MNIST  --sigma 32 --data_dir MNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResMNIST/FGSM/mnist_grayedge_32_robust_32.pth --net_type_substitute gray

num_epochs = opt.epochs # 20
batch_size = opt.batch_size # 100
attack_type = opt.attack #'FGSM'
net_type = opt.net_type #'grayedge'
data_dir = opt.data_dir #'MNIST'
which_model = opt.model #'mnist'
n_classes = opt.classes # 10
inp_size = opt.inp_size# 28
eps_t = opt.sigma
redetect_edge = opt.redetect_edge
net_type_substitute = opt.net_type_substitute
# eps_t = opt.eps



if not os.path.exists(f'./Res{data_dir}'):
    os.mkdir(f'./Res{data_dir}')

if not os.path.exists(f'./Res{data_dir}/{attack_type}'):
    os.mkdir(f'././Res{data_dir}/{attack_type}')


fo = open(f'./Res{data_dir}/{attack_type}/results_BPDA_{net_type}.txt', 'a+')

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
net, dataloader_dict_orig, _, _ = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
# if redetect_edge:
#     load_model(net, f'{opt.load_model}_{eps_t}_robust_{eps_t}_redetect.pth')
# else:
#     load_model(net, f'{opt.load_model}_{eps_t}_robust_{eps_t}.pth')
load_model(net, opt.load_model)
net.to(device)



save_path = f'{opt.load_model[:-4]}_BPDA.pth'    
substitue_net, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type_substitute, data_dir, inp_size, n_classes)
if os.path.exists(save_path):
    load_model(substitue_net, save_path)
else:
    train_substitue_model(net, substitue_net, dataloader_dict_orig, optimizer, num_epochs, save_path)
substitue_net.to(device)



# import pdb; pdb.set_trace()
acc_attack, _ = test_model_clean(substitue_net, dataloader_dict)
print('Accuracy of the substitute model on clean images: %f' % acc_attack)
fo.write('Accuracy of the substitute model on clean images: %f \n' % acc_attack)


acc_attack, _ = test_model_attack(substitue_net, dataloader_dict, epsilons, attack_type, net_type_substitute)
print('Accuracy of the substitute model on adversarial images: %f' % acc_attack[0])
fo.write('Accuracy of the substitute model on adversarial images: %f \n' % acc_attack[0])


acc_attack, _ = test_model_clean(net, dataloader_dict_orig)
print('Accuracy of the orig robust model on clean images: %f' % acc_attack)
fo.write('Accuracy of the orig robust model on clean images: %f \n' % acc_attack)


acc_attack, _ = test_model_attack(net, dataloader_dict_orig, epsilons, attack_type, net_type, redetect_edge=redetect_edge)
print('Accuracy of the orig robust model on adversarial images: %f' % acc_attack[0])
fo.write('Accuracy of the orig robust model on adversarial images: %f \n' % acc_attack[0])


acc_attack, _ = test_model_BPDA_attack(net, substitue_net, dataloader_dict_orig, epsilons, attack_type)
print('Accuracy of the orig robust model on BPDA adversarial images: %f' % acc_attack[0])
fo.write('Accuracy of the orig robust model on BPDA adversarial images: %f \n' % acc_attack[0])


fo.close()

