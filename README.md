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

+ Performance of the model against $L_\infty$ 
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


