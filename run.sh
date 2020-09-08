#!/bin/bash


# FASHION MNIST
# for sigma in 8 32 64
# python BPDA.py --net_type grayedge --model FashionMNIST  --data_dir FashionMNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResFashionMNIST/PGD/mnistfashion_grayedge_${sigma}_robust_${sigma}_redetect.pth --sigma ${sigma} --net_type_substitute gray --attack PGD

# for sigma in 8 32 64
# do
# 	python BPDA.py --net_type grayedge --model FashionMNIST  --data_dir FashionMNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResFashionMNIST/PGD/mnistfashion_grayedge_${sigma}_robust_${sigma}.pth --sigma ${sigma} --net_type_substitute gray --attack PGD
# done	



# # MNIST
# for sigma in 8 32 64
# do
# 	python BPDA.py --net_type grayedge --model MNIST  --data_dir MNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResMNIST/FGSM/mnist_grayedge_${sigma}_robust_${sigma}_redetect.pth --sigma ${sigma} --net_type_substitute gray # FGSM
# 	python BPDA.py --net_type grayedge --model MNIST  --data_dir MNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResMNIST/PGD/mnist_grayedge_${sigma}_robust_${sigma}_redetect.pth --sigma ${sigma} --net_type_substitute gray --attack PGD	
# done	




# # CIFAR
# for sigma in 8 32
# do
# 	# python BPDA.py --net_type rgbedge --model cifar10  --data_dir cifar10 --classes 10 --epochs 5 --inp_size 64 --load_model ResCIFAR10/FGSM/cifar10_rgbedge_${sigma}_robust_${sigma}.pth --sigma ${sigma} --net_type_substitute rgb # FGSM
# 	# python BPDA.py --net_type rgbedge --model cifar10  --data_dir cifar10 --classes 10 --epochs 5 --inp_size 64 --load_model ResCIFAR10/FGSM/cifar10_rgbedge_${sigma}_robust_${sigma}_redetect.pth --sigma ${sigma} --net_type_substitute rgb # FGSM
# 	python BPDA.py --net_type rgbedge --model cifar10  --data_dir cifar10 --classes 10 --epochs 5 --inp_size 64 --load_model  ResCIFAR10/PGD/cifar10_rgbedge_${sigma}_robust_${sigma}.pth --sigma ${sigma} --net_type_substitute rgb --attack PGD		
# 	python BPDA.py --net_type rgbedge --model cifar10  --data_dir cifar10 --classes 10 --epochs 5 --inp_size 64 --load_model  ResCIFAR10/PGD/cifar10_rgbedge_${sigma}_robust_${sigma}_redetect.pth --sigma ${sigma} --net_type_substitute rgb --attack PGD	
# done	



# python test_noise_robustness.py --n rgb -m icons_rgb.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n edge -m icons_edge.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n rgbedge -m icons_rgbedge.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n edge -m FGSM-Icons/icons_edge_8_robust_8.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n rgb -m FGSM-Icons/icons_rgb_8_robust_8.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n rgbedge -m FGSM-Icons/icons_rgbedge_8_robust_8.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n edge -m FGSM-Icons/icons_edge_32_robust_32.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n rgb -m FGSM-Icons/icons_rgb_32_robust_32.pth --data_dir Icons-50 --classes 50 --inp_size 64
# python test_noise_robustness.py --n rgbedge -m FGSM-Icons/icons_rgbedge_32_robust_32.pth --data_dir Icons-50 --classes 50 --inp_size 64




# some samples
# for training pix to pix
# python train_adv_gan.py --net_type gray --model mnist  --sigmas 8 32 --data_dir mnist --classes 10 --epochs 1 --inp_size 28 --gan_model netG_model_epoch_10.pth

# Training a robust classifier from the generated samples
# cd pix2pix-pytorch
# python train.py --niter 2 --niter_decay 1 --input_nc 1 --output_nc 1 --ngf 10 --ndf 10 --dataset mnist
# and testing it
# python test.py --dataset cifar10 --nepochs 60 --inp_size 32
# [you need to modify the ganutils.py file accordingly]





for S in 8 32 64
do
	# python test.py --net_type gray --model FashionMNIST  --sigma 0 --data_dir FashionMNIST --classes 10 --inp_size 28 --load_model ResFashionMNIST/FGSM/mnistfashion_edge_${S}_robust_${S}.pth --attack pgd	
	# python test.py --net_type gray --model FashionMNIST  --sigma $S --data_dir FashionMNIST --classes 10 --inp_size 28 --load_model ResFashionMNIST/FGSM/mnistfashion_edge_${S}_robust_${S}.pth --attack pgd
	python test.py --net_type gray --model MNIST  --sigma 0 --data_dir MNIST --classes 10 --inp_size 28 --load_model ResMNIST/FGSM/mnist_edge_${S}_robust_${S}.pth --attack pgd	
	python test.py --net_type gray --model MNIST  --sigma $S --data_dir MNIST --classes 10 --inp_size 28 --load_model ResMNIST/FGSM/mnist_edge_${S}_robust_${S}.pth --attack pgd

done	


# image-grid --folder ./sampleRobustness/ --n 75 --rows 15 --width 1000 --fill -bs 5