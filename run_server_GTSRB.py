from lib import *
from config import *
from model import build_model, build_model_resNet
from utils import * 
import torchattacks
from torchattacks import PGD, FGSM
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir
import torch.nn as nn
from torchvision import transforms, datasets, models


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NUM_EPOCHS = 10
BATCH_SIZE = 100


train_phase = True

attack_type = 'FGSM'

net_type = 'rgb'

data_dir = 'GTSRB'
inp_size = 64
n_classes = 50


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


fo = open(f'./{attack_type}-gtsrb/results/results_{net_type}.txt', 'w+')

# --------------------------------------------------------------------------------------------------------------------------------------------
# Train a model first
save_path = f'gtsrb_{net_type}.pth'


if train_phase:
    # pass
    net, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    net.to(device)
    train_model(net, dataloader_dict, criterior, optimizer, NUM_EPOCHS, save_path)



# NUM_EPOCHS = 30 # for adversarial training

# --------------------------------------------------------------------------------------------------------------------------------------------
# Test the clean model on clean and attacks
net, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
load_model(net, save_path)
net.to(device)

acc, images = test_model_clean(net, dataloader_dict)
print('Accuracy of original model on clean images: %f ' % acc)
fo.write('Accuracy of original model on clean images: %f \n' % acc)




for eps_t in [8,32]:

    print(f'eps_t={eps_t}')
    fo.write(f'eps_t={eps_t} \n')

    epsilons = [eps_t/255]


    # Test the clean model on clean and attacks
    net, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net, save_path)
    net.to(device)

    acc_attack, images = test_model_attack(net, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
    print('Accuracy of clean model on adversarial images: %f %%' % acc_attack[0])
    fo.write('Accuracy of clean model on adversarial images: %f \n' % acc_attack[0])


    net, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net, save_path)
    net.to(device)

    if net_type == 'rgbedge':
        acc_attack, images = test_model_attack(net, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
        print('Accuracy of clean model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
        fo.write('Accuracy of clean model on adversarial images with redetect_edge: %f \n' % acc_attack[0])


    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Now perform adversarial training
    save_path_robust = f'./{attack_type}-gtsrb/gtsrb_{net_type}_{eps_t}_robust_{eps_t}.pth'

    if train_phase:
        # pass    
        net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
        net_robust.to(device)
        train_robust_model(net_robust, dataloader_dict, criterior, optimizer, NUM_EPOCHS, save_path_robust, attack_type, eps=eps_t/255, net_type=net_type, redetect_edge=False)


    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Test the robust model on clean and attacks
    net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust) 
    net_robust.to(device)

    acc, images = test_model_clean(net_robust, dataloader_dict)
    print('Accuracy of robust model on clean images: %f %%' % acc)
    fo.write('Accuracy of robust model on clean images: %f \n' % acc)

    net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust)
    net_robust.to(device)

    acc_attack, images = test_model_attack(net_robust, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
    print('Accuracy of robust model on adversarial images: %f %%' % acc_attack[0])
    fo.write('Accuracy of robust model on adversarial images: %f \n' % acc_attack[0])


    net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust) 
    net_robust.to(device)
    
    if net_type == 'rgbedge':    
        acc_attack, images = test_model_attack(net_robust, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
        print('Accuracy of robust  model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
        fo.write('Accuracy of robust  model on adversarial images with redetect_edge: %f \n' % acc_attack[0])




    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Now perform adversarial training with redetect

    if net_type != 'rgbedge': continue

    save_path_robust = f'./{attack_type}-gtsrb/gtsrb_{net_type}_{eps_t}_robust_{eps_t}_redetect.pth'

    if train_phase:
        net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
        net_robust.to(device)
        train_robust_model(net_robust, dataloader_dict, criterior, optimizer, NUM_EPOCHS, save_path_robust, attack_type, eps=eps_t/255, net_type=net_type, redetect_edge=True)

    net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust) 
    net_robust.to(device)

    acc, images = test_model_clean(net_robust, dataloader_dict)
    print('Accuracy of robust redetect model on clean images: %f %%' % acc)
    fo.write('Accuracy of robust redetect model on clean images: %f \n' % acc)

    net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust) 
    net_robust.to(device)

    acc_attack, images = test_model_attack(net_robust, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
    print('Accuracy of robust redetect  model on adversarial images: %f %%' % acc_attack[0])
    fo.write('Accuracy of robust redetect  model on adversarial images: %f \n' % acc_attack[0])


    net_robust, dataloader_dict, criterior, optimizer = build_model_resNet(net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust)
    net_robust.to(device)
    
    acc_attack, images = test_model_attack(net_robust, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
    print('Accuracy of robust redtect model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
    fo.write('Accuracy of robust redetect model on adversarial images with redetect_edge: %f \n' % acc_attack[0])




fo.close()

