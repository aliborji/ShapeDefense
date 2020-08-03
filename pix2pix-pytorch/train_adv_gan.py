from __future__ import print_function
import sys
sys.path.insert(0,'..')



from lib import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
from gan_utils import *
from utils import *
from model import model_dispatcher



# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--epochs', type=int, default=20, help='num epochs')
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument('--attack', type=str, default='FGSM', help='attack type FGSM or PGD')
parser.add_argument('--net_type', type=str, default='rgb', help='edge, rgb, grayedge, rgbedge')
parser.add_argument('--data_dir', type=str, default='MNIST', help='data directory')
parser.add_argument('--model', type=str, default='mnist', help='which model implemented in model.py')
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')
parser.add_argument('--sigmas', nargs='+', type=int)
parser.add_argument('--gan_model', type=str, default='', help='path to the gan trained model')


# python train_adv_gan.py --net_type gray --model mnist  --sigmas 8 32 --data_dir mnist --classes 10 --epochs 1 --inp_size 28 --gan_model netG_model_epoch_10.pth
# python train_adv_gan.py --net_type gray --model mnist  --sigmas 8 32 --data_dir mnist --classes 10 --epochs 1 --inp_size 28 --gan_model netG_model_epoch_10.pth


opt = parser.parse_args()
print(opt)

num_epochs = opt.epochs # 20
batch_size = opt.batch_size # 100
attack_type = opt.attack #'FGSM'
net_type = opt.net_type #'grayedge'
data_dir = opt.data_dir #'MNIST'
which_model = opt.model #'mnist'
n_classes = opt.classes # 10
inp_size = opt.inp_size# 28
sigmas = opt.sigmas
gan_model = opt.gan_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))



fo = open(f'./result/results_{which_model}_{attack_type}.txt', 'a+')



# Load the already trained ga model on this dataset
model_path = "checkpoint/{}/{}".format(which_model, gan_model)
net_g = torch.load(model_path).to(device)



# Train a model now on generated images
save_path = "checkpoint/{}/{}_rob.pth".format(which_model, which_model)
_, dataloader_edge, _, _ = model_dispatcher(which_model, 'edge', data_dir, inp_size, n_classes)
net, _, criterion, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes) # send the actual images
if not os.path.exists(save_path):
	train_model_gan(net, dataloader_edge, criterion, optimizer, num_epochs, save_path, net_g)
# Test the clean model on clean and attacks
load_model(net, save_path)


# Test a model now on generated test data
acc = test_model_clean_gan(net, dataloader_edge, net_g) # send the edge maps
print('\nAccuracy of model on generated clean test images: %f ' % acc)
fo.write('Accuracy of model on generated clean test images: %f \n' % acc)


# # Test a model now on clean test set orginal data 
_, dataloader_img, _, _ = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes) # send the actual images
# acc = test_model_clean_gan(net, dataloader_img)      #### THIS DOES NOT MAKE SENSE
# print('Accuracy of model on original clean test images: %f ' % acc)
# fo.write('Accuracy of model on original clean test images: %f \n' % acc)


# _, dataloader_img, _, _ = model_dispatcher(which_model, 'rgbedge', data_dir, inp_size, n_classes) # send the actual images

for eps_t in opt.sigmas:
    print(f'eps_t={eps_t}')
    fo.write(f'eps_t={eps_t} \n')

    epsilon = eps_t/255


    acc_attack = test_model_attack_gan(net, dataloader_img, epsilon, attack_type, net_type, net_g)
    print('Accuracy of model on generated adversarial test images: %f ' % acc_attack)
    fo.write('Accuracy of model on generated adversarial test images: %f \n' % acc_attack)


    acc_attack = test_model_attack_gan(net, dataloader_img, epsilon, attack_type, net_type)
    print('Accuracy of model on original adversarial images: %f ' % acc_attack)
    fo.write('Accuracy of model on original adversarial images: %f \n' % acc_attack)


fo.close()
