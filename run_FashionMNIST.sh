#!/bin/bash


# FASHION MNIST
# for sigma in 8 32 64
# python BPDA.py --net_type grayedge --model FashionMNIST  --data_dir FashionMNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResFashionMNIST/PGD/mnistfashion_grayedge_${sigma}_robust_${sigma}_redetect.pth --sigma ${sigma} --net_type_substitute gray --attack PGD

# for sigma in 8 32 64
# do
# 	python BPDA.py --net_type grayedge --model FashionMNIST  --data_dir FashionMNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResFashionMNIST/PGD/mnistfashion_grayedge_${sigma}_robust_${sigma}.pth --sigma ${sigma} --net_type_substitute gray --attack PGD
# done	



# MNIST
for sigma in 8 32 64
do
	python BPDA.py --net_type grayedge --model FashionMNIST  --data_dir FashionMNIST --classes 10 --epochs 5 --inp_size 28 --load_model ResMNIST/PGD/mnistfashion_grayedge_${sigma}_robust_${sigma}.pth --sigma ${sigma} --net_type_substitute gray --attack PGD	
done	