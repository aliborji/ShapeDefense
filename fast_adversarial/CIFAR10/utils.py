import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from edge_detector import *
from skimage.filters import roberts, sobel, scharr
from config import *

#cifar10_mean = (0.4914, 0.4822, 0.4465)
#cifar10_std = (0.2471, 0.2435, 0.2616)

#mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
#std = torch.tensor(cifar10_std).view(3,1,1).cuda()

#upper_limit = ((1 - mu)/ std)
#lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    #import pdb ; pdb.set_trace()

#    global upper_limit
#    global lower_limit
#    global 
#    if X.shape[1]==1:
#        upper_limit = upper_limit.mean()
#        lower_limit = lower_limit.mean()
#        epsilon = epsilon.mean().reshape(1,1,1)
#        alpha = alpha.mean()
        #pgd_alpha = pgd_alpha.mean()

#    if X.shape[1] == 4:
#        upper_limit = torch.cat((upper_limit, upper_limit.mean().reshape(1,1,1)))
#        lower_limit = torch.cat((lower_limit, lower_limit.mean().reshape(1,1,1)))
#        epsilon = torch.cat((epsilon, epsilon.mean().reshape(1,1,1)))
#        alpha = torch.cat((alpha, alpha.mean().reshape(1,1,1)))






    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, net_type='rgb', redetect=False):
#    epsilon = (8 / 255.) / std
#    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()


#    global upper_limit
#    global  lower_limit
    #if net_type == 'edge':
    #    upper_limit = upper_limit.mean()
    #    lower_limit = lower_limit.mean()
    #    epsilon = epsilon.mean().reshape(1,1,1)
    #    alpha = alpha.mean()
    #    #pgd_alpha = pgd_alpha.mean()

    #if net_type == 'rgbedge':
    #    upper_limit = torch.cat((upper_limit, upper_limit.mean().reshape(1,1,1)))
    #    lower_limit = torch.cat((lower_limit, lower_limit.mean().reshape(1,1,1)))
    #    epsilon = torch.cat((epsilon, epsilon.mean().reshape(1,1,1)))
    #    alpha = torch.cat((alpha, alpha.mean().reshape(1,1,1)))




    for i, (X, y) in enumerate(test_loader):
        if net_type.lower() in ['rgbedge', 'edge']: 
            X = detect_edge_batch(X);
            if net_type.lower() == 'edge':
                X = X[:,-1].unsqueeze(1)


        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            # redetect edge
            Z = X + pgd_delta
            if net_type.lower() == 'rgbedge' and redetect:
                Z = Z[:,:3,...]
                Z = Z.cpu()
                Z = detect_edge_batch(Z) #[:,:3,...])
                Z = Z.cuda()
            
            output = model(Z)
	   	
#            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model, net_type='rgb'):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if net_type.lower() in ['rgbedge', 'edge']: # and redetect_edge: # for 
             X = detect_edge_batch(X);
             if net_type.lower() == 'edge':
                X= X[:,-1].unsqueeze(1)

            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def detect_edge_batch(imgs):        
        ZZ = torch.zeros((imgs.shape[0], imgs.shape[1]+1, imgs.shape[2], imgs.shape[3]))  
        ZZ[:,:3,...] = imgs
        for idx, im in enumerate(imgs):
            # edge_map = detect_edge_new(im.permute(1,2,0)) # make it XxYx3!!!
            img = im.permute(1,2,0) # make it XxYx3!!!            
            # gray = np.array(img.mean(axis=2)*255).astype('uint8')
            # edge_map = roberts(gray)
            # edge_map = edge_map/255.
            # edge_map = detect_edge_new_cifar(img[:,:,:3])
            edge_map = detect_edge_tiny(img[:,:,:3])            
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            # im = torch.cat((im, edge_map[None]),dim=0)     
            # import pdb; pdb.set_trace()
            ZZ[idx,-1] = edge_map
        imgs = ZZ    
        return imgs



def detect_edge_new_cifar(img): 
  import pdb; pdb.set_trace()
  gray = np.array(img.mean(axis=2)*255).astype('uint8')
  imgBLR = cv2.GaussianBlur(gray, (5,5), 2)
  imgEDG = cv2.Canny(imgBLR, 40, 150)  
  if (imgEDG.max() - imgEDG.min()) > 0:
    imgEDG = (imgEDG - imgEDG.min()) / (imgEDG.max() - imgEDG.min())
  # imgEDG = imgEDG/255.
  return imgEDG




def detect_edge_tiny(img):
  gray = np.array(img.mean(axis=2).cpu()*1).astype('float64')
  edge_map = feature.canny(gray, sigma = 2)#, low_threshold=0.01, high_threshold=.2)
  # edge_map = edge_map/255.

  return edge_map







