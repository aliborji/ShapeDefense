# Shape Defense



### Prerequisites

My OS: **Ubuntu 18.04**

Environment: **Anaconda**, **Python 3.6**

### Installing

```bash
conda create -y --name py36 python==3.6
conda install -f -y -q --name py36 -c conda-forge --file requirements.txt
conda activate py36
git clone https://github.com/baoanh1310/dogs_vs_cats_pytorch.git
cd dogs_vs_cats_pytorch
```

### Running

Use Google Colab to run **training.ipynb** notebook.

You can change NUM_EPOCHS in **config.py** module to train more/less epochs on your machine.

My pretrained model which achieve 70%+ accuracy on the dataset is the **dogs_cats.pth** file.
