from lib import *
from torchvision import datasets
from torchvision import transforms, datasets, models
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from config import *
from edge_detector import *


class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train', net_type='rgb'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.net_type = net_type        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        if self.net_type == 'rgb':
            img_transformed = self.transform(img, self.phase)
            img_transformed = img_transformed - img_transformed.min()
            
        elif self.net_type == 'edge': # rgb_egde
            img_transformed = self.transform(img, self.phase)            
            edge_map = edge_detect(img_transformed.permute(1,2,0))
            # edge_map = edge_map/255.
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img_transformed = edge_map[None]
        
        else: # rgb + edge
            # borji
            img_transformed = self.transform(img, self.phase)            
            edge_map = edge_detect(img_transformed.permute(1,2,0))
            # edge_map = edge_map/255.
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img_transformed = torch.cat((img_transformed, edge_map[None]),dim=0)      
        
        
        label = img_path.split('/')[-2]

        if label == 'dogs':
            label = 0
        elif label == 'cats':
            label = 1

        return img_transformed, label





class Dataset_MNIST(data.Dataset):
    def __init__(self, transform=None, phase='train', net_type='gray'):
        self.data = datasets.MNIST('../data', train = phase=='train', download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        ]))        
        
        self.data = datasets.MNIST('../data', train = phase=='train', download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        ]))        
#         self.transform = transform
        self.phase = phase
        self.net_type = net_type        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         import pdb; pdb.set_trace()

        img, label = self.data[idx][0], self.data[idx][1]
    
        if self.net_type == 'gray':
#             img_transformed = self.transform(img, self.phase)
#             img_transformed = img_transformed - img_transformed.min()
              pass 
            
        elif self.net_type == 'edge': # rgb_egde
#             img_transformed = self.transform(img, self.phase)            
            edge_map = edge_detect(img)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = edge_map#[None]
        
        else: # gray + edge
            # borji
            edge_map = edge_detect(img)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())            
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = torch.cat((img, edge_map),dim=0)#[None]
        


        return img, label



class Dataset_CIFAR10(data.Dataset):
    def __init__(self, transform=None, phase='train', net_type='rgb'):
        self.data = datasets.CIFAR10('../data', train = phase=='train', download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        ]))        
        
        self.data = datasets.CIFAR10('../data', train = phase=='train', download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        ]))        
#         self.transform = transform
        self.phase = phase
        self.net_type = net_type        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         import pdb; pdb.set_trace()

        img, label = self.data[idx][0], self.data[idx][1]
    
        if self.net_type == 'rgb':
#             img_transformed = self.transform(img, self.phase)
#             img_transformed = img_transformed - img_transformed.min()
              pass 
            
        elif self.net_type == 'edge': # rgb_egde
#             img_transformed = self.transform(img, self.phase)            
            edge_map = edge_detect(img)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = edge_map#[None]
        
        else: # gray + edge
            # borji
            edge_map = edge_detect(img)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())            
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = torch.cat((img, edge_map),dim=0)#[None]
        


        return img, label

    

    
class Dataset_FashionMNIST(data.Dataset):
    def __init__(self, transform=None, phase='train', net_type='gray'):
        self.data = datasets.FashionMNIST('../data', train = phase=='train', download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        ]))        
        
        self.data = datasets.FashionMNIST('../data', train = phase=='train', download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        ]))        
#         self.transform = transform
        self.phase = phase
        self.net_type = net_type        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img, label = self.data[idx][0], self.data[idx][1]
    
        if self.net_type == 'gray':
#             img_transformed = self.transform(img, self.phase)
#             img_transformed = img_transformed - img_transformed.min()
              pass 
            
        elif self.net_type == 'edge': # rgb_egde
            edge_map = edge_detect(img)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = edge_map#[None]
        
        else: # gray + edge
            edge_map = edge_detect(img)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())            
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = torch.cat((img, edge_map), dim=0)#[None]
        


        return img, label

    

class DogsDataset(data.Dataset):
    # for dog breed classification
    def __init__(self, labels, root_dir, edge_dir, subset=False, transform=None, net_type='rgb', inp_size=None):
        self.labels = labels
        self.root_dir = root_dir
        self.edge_dir = edge_dir        
        self.transform = transform
        self.net_type = net_type                
#         self.phase = phase                        
    
        self.edge_transform = transforms.Compose([transforms.Scale(inp_size),
                                   transforms.CenterCrop(inp_size),
                                   transforms.ToTensor()])    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.net_type == 'rgb':
            img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
            fullname = osp.join(self.root_dir, img_name)
            img = Image.open(fullname)
            img_transformed = self.transform(img)
#             img_transformed = img_transformed - img_transformed.min()
            
        elif self.net_type == 'edge': # edge; already precomputed!
            img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
            fullname = osp.join(self.edge_dir, img_name)
            edge_map = Image.open(fullname)
            img_transformed = self.edge_transform(edge_map)
                    
        else: # rgb + edge
            img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
            
            fullname = osp.join(self.root_dir, img_name)
            image = Image.open(fullname)
            img = self.transform(image)

            fullname = osp.join(self.edge_dir, img_name)
            edge_map = Image.open(fullname)
            edge_map = self.edge_transform(edge_map)
            
            img_transformed = torch.cat((img, edge_map),dim=0)              
        
        
        labels = self.labels.iloc[idx, 1:]#.as_matrix().astype('float')
        labels = np.argmax(labels)
        return [img_transformed, labels]        




class folderDB(Dataset):

    def __init__(self, root_dir, train=False, transform=None, net_type='rgb', base_folder=''):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.base_folder = base_folder


        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

        self.net_type = net_type

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)


        if self.net_type == 'rgb':
            pass            

        elif self.net_type == 'gray':
            img = img.mean(axis=0).unsqueeze(0) #input has three channels but is gray level; \eg sketch dataset
            img = (img - img.min()) / (img.max() - img.min())                    
        
        elif self.net_type == 'edge': # rgb_egde
            edge_map = edge_detect(img)
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            if (edge_map.max() - edge_map.min()) > 0:
                edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        

            img = edge_map[None]


        elif self.net_type == 'grayedge': # notice input has three channels here! but is gray level! sketch db; all pixel ranges normalized to [0 1]
            edge_map = edge_detect(img)
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            if (edge_map.max() - edge_map.min()) > 0:
                edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        

            grayImg = img.mean(axis=0).unsqueeze(0) #input has three channels; in case of sketch dataset
            grayImg = (grayImg - grayImg.min()) / (grayImg.max() - grayImg.min())                                

            img = torch.cat((grayImg, edge_map[None]),dim=0)      


        else: # rgb + edge
            edge_map = edge_detect(img)
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            if (edge_map.max() - edge_map.min()) > 0:
                edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        

            img = torch.cat((img, edge_map[None]),dim=0)      


        classId = self.csv_data.iloc[idx, 1]
        
        return img, classId



