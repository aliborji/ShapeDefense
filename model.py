from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model, load_model
from dataset import MyDataset, Dataset_MNIST, Dataset_FashionMNIST, DogsDataset, folderDB, Dataset_CIFAR10
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os.path import isfile, join, abspath, exists, isdir, expanduser
import os
from torch.optim import lr_scheduler
from time import time


# Binary network used for dog vs cat classification
class Net(nn.Module):
    def __init__(self, net_type):
        super(Net, self).__init__()
        if net_type == 'rgb':
            self.conv1 = nn.Conv2d(3, 32, 3)        
        elif net_type == 'edge': # rgb_egde
            self.conv1 = nn.Conv2d(1, 32, 3)        
        else: # rgb + edge
            self.conv1 = nn.Conv2d(4, 32, 3)  

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*7*7, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def build_model(net_type):
#     import pdb; pdb.set_trace()
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # Create dataset objects
    train_dataset = MyDataset(train_list, ImageTransform((HEIGHT, WIDTH), MEAN, STD), phase='train', net_type=net_type)
    val_dataset = MyDataset(val_list, ImageTransform((HEIGHT, WIDTH), MEAN, STD), phase='val', net_type=net_type)

    # Create dataloader objects
    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Build model
    net = Net(net_type)
    print(net)


    # Loss 
    criterior = nn.CrossEntropyLoss()

    # Optimizer
    params = net.parameters()
#     optimizer = optim.RMSprop(params, lr=1e-4)
    optimizer = optim.Adam(params, lr=1e-4)    

    return net, dataloader_dict, criterior, optimizer




# LeNet Model definition
class MNIST_Net(nn.Module):
    def __init__(self, net_type):
        super(MNIST_Net, self).__init__()
        if net_type in ['gray', 'edge']:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        else: # gray + edge
            self.conv1 = nn.Conv2d(2, 10, kernel_size=5)

        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x#F.log_softmax(x, dim=1)



def build_model_mnist(net_type):
#     import pdb; pdb.set_trace()
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # Create dataset objects
    train_dataset = Dataset_MNIST(phase='train', net_type=net_type)
    val_dataset = Dataset_MNIST(phase='val', net_type=net_type)

    # Create dataloader objects

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100 , shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Build model
    net = MNIST_Net(net_type)
    print(net)


    # Loss 
    criterior = nn.CrossEntropyLoss()

    # Optimizer
    params = net.parameters()
#     optimizer = optim.RMSprop(params, lr=1e-4)
    optimizer = optim.Adam(params, lr=1e-4)    

    # Training model
    # train_model(net, dataloader_dict, criterior, optimizer, NUM_EPOCHS)
    return net, dataloader_dict, criterior, optimizer




def build_model_fashion_mnist(net_type):
#     import pdb; pdb.set_trace()
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # Create dataset objects
    train_dataset = Dataset_FashionMNIST(phase='train', net_type=net_type)
    val_dataset = Dataset_FashionMNIST(phase='val', net_type=net_type)

    # Create dataloader objects
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Build model
    net = MNIST_Net(net_type)
    print(net)


    # Loss 
    criterior = nn.CrossEntropyLoss()

    # Optimizer
    params = net.parameters()
#     optimizer = optim.RMSprop(params, lr=1e-4)
    optimizer = optim.Adam(params, lr=1e-4)    

    # Training model
    # train_model(net, dataloader_dict, criterior, optimizer, NUM_EPOCHS)
    return net, dataloader_dict, criterior, optimizer





def build_model_dogs(net_type, data_dir, inp_size):

    INPUT_SIZE = inp_size
    NUM_CLASSES = 16
#     data_dir = './dog-breed-identification/'
    labels = pd.read_csv(osp.join(data_dir, 'labels.csv'))
    sample_submission = pd.read_csv(osp.join(data_dir, 'sample_submission.csv'))
    print(len(os.listdir(osp.join(data_dir, 'train'))), len(labels))

#     import pdb; pdb.set_trace()
    selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
    labels = labels[labels['breed'].isin(selected_breed_list)]
    labels['target'] = 1
    # labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)


    num_train = int(labels_pivot.shape[0]*.8)
    train = labels_pivot[:num_train]# .sample(frac=0.8)    
    valid = labels_pivot[num_train:]

    # import pdb; pdb.set_trace()
    # train = labels_pivot.sample(frac=0.8)    
    # valid = labels_pivot[~labels_pivot['id'].isin(train['id'])]

    print(train.shape, valid.shape)


    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    # ds_trans = transforms.Compose([transforms.Scale(224),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                normalize])

    ds_trans = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    train_ds = DogsDataset(train, data_dir+'traincrop/', data_dir+'traincropedgesobel/', transform=ds_trans, net_type=net_type, inp_size=INPUT_SIZE)
    valid_ds = DogsDataset(valid, data_dir+'traincrop/', data_dir+'traincropedgesobel/', transform=ds_trans, net_type=net_type, inp_size=INPUT_SIZE)
    # test is over the validation set

    train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(valid_ds, batch_size=4, shuffle=True, num_workers=4)


    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}


    # Build model
    resnet = models.resnet18(pretrained=False)
    # resnet = models.resnet18(pretrained=True)  
    resnet.load_state_dict(torch.load(data_dir + '/resnet18-5c106cde.pth')) #resnet50-19c8e357.pth'))

    # import pdb; pdb.set_trace()
    if net_type == 'rgb':
        pass
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
    elif net_type == 'edge': # rgb_egde
        with torch.no_grad():
          new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,0] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer
    else: # rgb + edge
        with torch.no_grad():
          new_layer = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,:3] = resnet.conv1.weight.squeeze(1) 
          new_layer.weight[:,3] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer

        # resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    

    
    
    # freeze all model parameters
    # for param in resnet.parameters():
    #     param.requires_grad = False

    resnet.conv1.requires_grad = True

    # new final layer with 16 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD([resnet.fc.parameters(), resnet.conv1.weight], lr=0.001, momentum=0.9)

    # YOU MAY WANT TO OPTIMIZE ALL PARAMETERS BY USING resnet.parameters() 
    # optimizer = torch.optim.SGD(list(resnet.fc.parameters()) + list(resnet.conv1.parameters()), lr=0.001, momentum=0.9)

    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # print(resnet)
    print('Model loaded ...')


    return resnet, dataloader_dict, criterion, optimizer#, exp_lr_scheduler   






def build_model_resNet(net_type, data_dir, inp_size, n_classes):
    # import pdb; pdb.set_trace()

    INPUT_SIZE = inp_size
    NUM_CLASSES = n_classes


    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])



    trainset = folderDB(
        root_dir='.', train=True,  transform=transform, net_type=net_type, base_folder=data_dir)
    testset = folderDB(
        root_dir='.', train=False,  transform=transform, net_type=net_type, base_folder=data_dir)


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)



    dataloader_dict = {'train': trainloader, 'val': testloader}


    # Build model
    resnet = models.resnet18(pretrained=True)

    # import pdb; pdb.set_trace()
    if net_type == 'rgb':
          # new_layer = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # resnet.conv1 = new_layer
        pass  

        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
    elif net_type in ['edge', 'gray']: # rgb_egde
        with torch.no_grad():
          new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,0] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer

    elif net_type == 'grayedge': # rgb + edge
        with torch.no_grad():
          new_layer = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,0] = torch.mean(resnet.conv1.weight, 1)#[:,None]
          new_layer.weight[:,1] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer


    else: # rgb + edge
        with torch.no_grad():
          new_layer = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,:3] = resnet.conv1.weight.squeeze(1) 
          new_layer.weight[:,3] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer

        # resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    

    
    
    # freeze all model parameters
    # for param in resnet.parameters():
    #     param.requires_grad = False

    resnet.conv1.requires_grad = True

    # new final layer with 16 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD([resnet.fc.parameters(), resnet.conv1.weight], lr=0.001, momentum=0.9)
    
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    

    # optimizer = torch.optim.SGD(list(resnet.fc.parameters()) + list(resnet.conv1.parameters()), lr=0.001, momentum=0.9)    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # print(resnet)
    print('Model loaded ...')


    return resnet, dataloader_dict, criterion, optimizer#, exp_lr_scheduler   






def build_model_resNet_CIFAR10(net_type, data_dir, inp_size, n_classes):
    # import pdb; pdb.set_trace()

    INPUT_SIZE = inp_size
    NUM_CLASSES = 10


    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])


#     import pdb; pdb.set_trace()
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # Create dataset objects
    train_dataset = Dataset_CIFAR10(phase='train', net_type=net_type, transform=transform)
    val_dataset = Dataset_CIFAR10(phase='val', net_type=net_type, transform=transform)

    # Create dataloader objects
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}



    # Build model
    resnet = models.resnet18(pretrained=True)

    # import pdb; pdb.set_trace()
    if net_type == 'rgb':
          # new_layer = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # resnet.conv1 = new_layer
        pass  

        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
    elif net_type in ['edge', 'gray']: # rgb_egde
        with torch.no_grad():
          new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,0] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer

    elif net_type == 'grayedge': # rgb + edge
        with torch.no_grad():
          new_layer = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,0] = torch.mean(resnet.conv1.weight, 1)#[:,None]
          new_layer.weight[:,1] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer


    else: # rgb + edge
        with torch.no_grad():
          new_layer = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # new_layer.requires_grad = False
          new_layer.weight[:,:3] = resnet.conv1.weight.squeeze(1) 
          new_layer.weight[:,3] = torch.mean(resnet.conv1.weight, 1)#[:,None]
         # resnet.conv1.weight = new_layer #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          resnet.conv1 = new_layer

        # resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    

    
    
    # freeze all model parameters
    # for param in resnet.parameters():
    #     param.requires_grad = False

    resnet.conv1.requires_grad = True

    # new final layer with 16 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD([resnet.fc.parameters(), resnet.conv1.weight], lr=0.001, momentum=0.9)
    
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    

    # optimizer = torch.optim.SGD(list(resnet.fc.parameters()) + list(resnet.conv1.parameters()), lr=0.001, momentum=0.9)    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # print(resnet)
    print('Model loaded ...')


    return resnet, dataloader_dict, criterion, optimizer#, exp_lr_scheduler   

