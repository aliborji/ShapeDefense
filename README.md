# Shape Defense

* This repository includes the following:
+ Edge-guided Adversarial Training (EAT)
+ GAN-based Shape Defense (GSD)
+ Shape-based fast adversarial training
+ Shape-based free adversarial training
+ Robustness evaluation against natural image corruptions

It shows that incorporating edges to images followed by adversarial training and edge redetection at inference time improves robustness drastically.

![overfitting](https://github.com/aliborji/ShapeDefence/blob/master/teaser.jpg)


[paper]: https://arxiv.org/abs/xx
[fastpaper]: https://arxiv.org/abs/2001.03994


## News
+ 12/19/2019 - TBD

## Installation 
1. Installation
2. Install PyTorch.
Install the required python packages. All packages can be installed by running the following command:
```
pip install -r requirements.txt
```
3. For each of the code repositories used, please see the accompanying readme files and original code bases.


##  Training and evaluating a model
To train a robust model run the following command:

```
python train.py --net_type [NET_TYPE]

 --model [MODEL_TYPE] --sigmas [LIST_OF_SIGMAS]  --data_dir [DATA_FOLDER]  --classes 10 --epochs 10 --inp_size 28 --load_model [LOAD_MODEL_IF_EXISTS_RESUME]
```

rgbedge
fashionmnist
8 32 64
fashionmnist


This trains a robust model with the default parameters. The training parameters can be set by changing the configs.yml config file. Please run python main_free.py --help to see the list of possible arguments. The script saves the trained models into the trained_models folder and the logs into the output folder.




## Results over MNIST and CIFAR-10

+ Performance of the model against $L_\inf$ 
Please see the paper for more details.

|          | CIFAR-10 Acc | CIFAR10 Adv Acc (eps=8/255) | Time (minutes) | 
| --------:| -----------:|----------------------------:|---------------:| 
| FGSM     |      86.06% |                      46.06% |             12 |
| Free     |      85.96% |                      46.33% |            785 |
| PGD      |      87.30% |                      45.80% |           4966 |

|          | ImageNet Acc | ImageNet Adv Acc (eps=2/255) | Time (hours) | 
| --------:| ------------:|-----------------------------:|-------------:| 
| FGSM     |       60.90% |                       43.46% |           12 |
| Free     |       64.37% |                       43.31% |           52 |

## But I've tried FGSM adversarial training before, and it didn't work! 
In our experiments, we discovered several failure modes which would cause FGSM adversarial training to ``catastrophically fail'', like in the following plot. 



If FGSM adversarial training hasn't worked for you in the past, then it may be because of one of the following reasons (which we present as a non-exhaustive list of ways to fail): 

+ FGSM step size is too large, forcing the adversarial examples to cluster near the boundary
+ Random initialization only covers a smaller subset of the threat model
+ Long training with many epochs and fine tuning with very small learning rates

All of these pitfalls can be avoided by simply using early stopping based on a subset of the training data to evaluate the robust accuracy with respect to PGD, as the failure mode for FGSM adversarial training occurs quite rapidly (going to 0% robust accuracy within the span of a couple epochs)



## Why does this matter if I still want to use PGD adversarial training in my experiments? 

The speedups gained from using mixed-precision arithmetic and cyclic learning rates can still be reaped regardless of what training regimen you end up using! For example, these techniques can speed up CIFAR10 PGD adversarial training by almost 2 orders of magnitude, reducing training time by about 3.5 days to just over 1 hour. The engineering costs of installing the `apex` library and changing the learning rate schedule are miniscule in comparison to the time saved from using these two techniques, and so even if you don't use FGSM adversarial training, you can still benefit from faster experimentation with the DAWNBench improvements. 



## Citation

If you use this code in your research, please cite this project.

```
@article{reluDefense2020,
  title={Harnessing adversarial examples with a surprisingly simple defense},
  author={Borji, Al},
  journal={arXiv preprint arXiv:2004.13013},
  year={2020}
}
```


