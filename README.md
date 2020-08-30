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
1. Install PyTorch.
2. Install the required python packages. All packages can be installed by running the following command:
```
pip install -r requirements.txt
```
3. For each of the code repositories used, please see the accompanying readme files and original code bases.


##  Training and evaluating a model
To train a robust model run the following command:

```
python train.py --net_type [NET_TYPE] --model [MODEL_TYPE] --sigmas [LIST_OF_SIGMAS]  --data_dir [DATA_FOLDER]  --classes 10 --epochs 10 --inp_size 28 --load_model [LOAD_MODEL_IF_EXISTS]
```

This trains a robust model with the following parameters. [NET_TYPE] can be edge, rgb or rgbedge. [MODEL_TYPE] determines the type of the model and [DATA_FOLDER] sets the folder containing data. Please see model.py for some examples. [LIST_OF_SIGMAS] is the list of perturbation budgets. It trains one several models for each sigma. [LOAD_MODEL_IF_EXISTS] loads a trained classifier to build a robust model for it. See sample_pgd40_mnist_results folder for some sample generated results.
You also need to specify the type of the edge detector that you want to use in the config.py.
For some other codes, I have placed a sample line how to call it at the begining of the .py file. Also please see run.sh for running the code in scale.


### Additional:
+ Run BPDA.py if you want to conduct a substitute attack! (bad naming here perhaps!)
+ Run test_noise_robustness.py for testing a model against natural perturbations.
+ Run edge_only_attack.py if you want to only attack ede pixels (i.e., removing attack on non-edge pixels)
+ Run plots.ipynb for plotting results
+ See also visualization for runing edge detection on clean and attacked images



## Citation

If you use this code in your research, please cite this project.

```
@article{shapeDefense2020,
  title={Shape Defense},
  author={Borji, Al},
  journal={xx},
  year={2020}
}
```


