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



python test_noise_robustness.py --n rgb -m icons_rgb.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n edge -m icons_edge.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n rgbedge -m icons_rgbedge.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n edge -m FGSM-Icons/icons_edge_8_robust_8.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n rgb -m FGSM-Icons/icons_rgb_8_robust_8.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n rgbedge -m FGSM-Icons/icons_rgbedge_8_robust_8.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n edge -m FGSM-Icons/icons_edge_32_robust_32.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n rgb -m FGSM-Icons/icons_rgb_32_robust_32.pth --data_dir Icons-50 --classes 50 --inp_size 64
python test_noise_robustness.py --n rgbedge -m FGSM-Icons/icons_rgbedge_32_robust_32.pth --data_dir Icons-50 --classes 50 --inp_size 64
