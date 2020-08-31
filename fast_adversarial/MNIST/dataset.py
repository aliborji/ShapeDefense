from lib import *
from edge_detector import *
from torchvision import datasets
from torchvision import transforms, datasets, models

# class MyDatasetRGB(data.Dataset):
#     def __init__(self, file_list, transform=None, phase='train'):
#         self.file_list = file_list
#         self.transform = transform
#         self.phase = phase

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         img_path = self.file_list[idx]
#         img = Image.open(img_path)

#         img_transform = self.transform(img, self.phase)
        
        
#         label = img_path.split('/')[-2]

#         if label == 'dogs':
#             label = 0
#         elif label == 'cats':
#             label = 1

#         return img_transformed, label    



class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train', net_type='rgb'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.net_type = net_type        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
#         import pdb; pdb.set_trace()
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        if self.net_type.lower() == 'rgb':
            img_transformed = self.transform(img, self.phase)
            img_transformed = img_transformed - img_transformed.min()
            
        elif self.net_type.lower() == 'edge': # rgb_egde
            img_transformed = self.transform(img, self.phase)            
            edge_map = detect_edge_new(img_transformed.permute(1,2,0))
            edge_map = edge_map/255.
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img_transformed = edge_map[None]
        
        else: # rgb + edge
            # borji
            img_transformed = self.transform(img, self.phase)            
            edge_map = detect_edge_new(img_transformed.permute(1,2,0))
            edge_map = edge_map/255.
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
    
        if self.net_type.lower() == 'gray':
#             img_transformed = self.transform(img, self.phase)
#             img_transformed = img_transformed - img_transformed.min()
              pass 
            
        elif self.net_type.lower() == 'edge': # rgb_egde
#             img_transformed = self.transform(img, self.phase)            
            edge_map = detect_edge_mnist(img)
            edge_map = edge_map/255.
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = edge_map#[None]
        
        else: # gray + edge
            # borji
            edge_map = detect_edge_mnist(img)
            edge_map = edge_map/255.
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
#         import pdb; pdb.set_trace()

        img, label = self.data[idx][0], self.data[idx][1]
    
        if self.net_type.lower() == 'gray':
#             img_transformed = self.transform(img, self.phase)
#             img_transformed = img_transformed - img_transformed.min()
              pass 
            
        elif self.net_type.lower() == 'edge': # rgb_egde
#             img_transformed = self.transform(img, self.phase)            
            edge_map = detect_edge_mnist(img)
            edge_map = edge_map/255.
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = edge_map#[None]
        
        else: # gray + edge
            # borji
            edge_map = detect_edge_mnist(img)
            edge_map = edge_map/255.
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())            
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            img = torch.cat((img, edge_map),dim=0)#[None]
        


        return img, label

    

class DogsDataset(data.Dataset):
    def __init__(self, labels, root_dir, edge_dir, subset=False, transform=None, net_type='rgb'):
        self.labels = labels
        self.root_dir = root_dir
        self.edge_dir = edge_dir        
        self.transform = transform
        self.net_type = net_type                
#         self.phase = phase                        
    
        self.edge_transform = transforms.Compose([transforms.Scale(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()])    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.net_type.lower() == 'rgb':
            img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
            fullname = osp.join(self.root_dir, img_name)
            img = Image.open(fullname)
            img_transformed = self.transform(img)
#             img_transformed = img_transformed - img_transformed.min()
            
        elif self.net_type.lower() == 'edge': # edge
#             import pdb; pdb.set_trace()
            img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
            fullname = osp.join(self.edge_dir, img_name)
            edge_map = Image.open(fullname)
            img_transformed = self.edge_transform(edge_map)/255.            
                    
        else: # rgb + edge
            img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
            
            fullname = osp.join(self.root_dir, img_name)
            image = Image.open(fullname)
            img = self.transform(img, self.phase)

            fullname = osp.join(self.edge_dir, img_name)
            edge_map = Image.open(fullname)
            edge_map = self.edge_transform(edge_map)/255.            
            
            img_transformed = torch.cat((img, edge_map[None]),dim=0)              
        
        
        labels = self.labels.iloc[idx, 1:]#.as_matrix().astype('float')
        labels = np.argmax(labels)
        return [img_transformed, labels]        