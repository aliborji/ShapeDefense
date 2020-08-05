# Shape Defense Pytorch

In this project, I use Pytorch to implement a very simple VGG-based CNN model to classify dog/cat images of Kaggle's Dogs vs. Cats Dataset.
https://guides.github.com/activities/hello-world/

### Installing

```bash
conda create -y --name py36 python==3.6
conda install -f -y -q --name py36 -c conda-forge --file requirements.txt
conda activate py36
git clone https://github.com/aliborji/ShapeDefence.git
cd ShapeDefence
```

### Running

Use Google Colab to run **training.ipynb** notebook.

You can change NUM_EPOCHS in **config.py** module to train more/less epochs on your machine.

My pretrained model which achieve 70%+ accuracy on the dataset is the **dogs_cats.pth** file.
