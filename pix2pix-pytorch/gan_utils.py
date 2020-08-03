from lib import *
from config import *
# import copy
import torchattacks
from torchattacks import PGD, FGSM
import time
import numpy as np
from PIL import Image
from utils import detect_edge_batch


# from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate

def train_model_gan(net, dataloader_dict, criterior, optimizer, num_epochs, save_path, net_g):

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    phase = 'train'
    # boost network speed on gpu
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        net.train()

        epoch_loss = 0.0
        epoch_corrects = 0

        for inputs, labels in tqdm(dataloader_dict[phase]):
            # import pdb; pdb.set_trace()
            inputs = inputs.to(device)
            inputs = net_g(inputs) # pass the inputs through the GAN
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                labels = labels
                loss = criterior(outputs, labels)
                _, preds = torch.max(outputs, axis=1)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.shape[0]
                epoch_corrects += torch.sum(preds==labels.data)


        epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
        epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

        print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict(), save_path)



def test_model_clean_gan(net, dataloader_dict, net_g=None):
    correct = 0
    total = 0

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    net.eval()
    
    
    for inputs, labels in dataloader_dict['val']:    
        # images = (images-images.min()) / (images.max()-images.min())
        inputs = inputs.to(device)

        # import pdb; pdb.set_trace()
        if net_g:
            inputs = net_g(inputs) # pass the inputs through the GAN
        
        labels = labels.to(device)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
    #     correct += (predicted == labels.cuda()).sum()
        correct += (predicted == labels).sum()

    acc = float(correct) / total
        
    return acc    


def test_model_attack_gan(net, dataloader_dict, eps, attack_type = 'FGSM', net_type='rgb', net_g=None):
    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    net.eval()
    # accuracies = []; examples = []

    if attack_type.upper() == 'FGSM':
        attack = FGSM(net, eps=eps)    
    else:    
        attack = PGD(net, eps=eps, alpha=8/255, iters=40)      
    
        
    correct = 0; total = 0;       

    for images, labels in dataloader_dict['val']:

        images = images.to(device)
        labels = labels.to(device)




        # import pdb; pdb.set_trace()

        images = attack(images.detach(), labels)#.cuda()
        # thresholding
        # images[images<=eps] = 0
        # import pdb; pdb.set_trace()    

        if net_g:
            # import pdb; pdb.set_trace()    
            edge_maps = torch.zeros((images.shape[0],1,images.shape[2],images.shape[2]))
            images = torch.cat((images, edge_maps),dim=1)#[None]
            images = detect_edge_batch(images)
            # import pdb; pdb.set_trace()    
            images = net_g(images[:,-1].unsqueeze(1)) # pass the images through the GAN

        # import pdb; pdb.set_trace()    
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = float(correct) / total
    # accuracies.append(acc)

    return acc#accuracies    




def perform_attack(net, images, labels, eps, attack_type = 'FGSM', net_type='rgb'):
    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    net.eval()
    # accuracies = []; examples = []

    if attack_type.upper() == 'FGSM':
        attack = FGSM(net, eps=eps)    
    else:    
        attack = PGD(net, eps=eps, alpha=8/255, iters=40)      
    
        

    images = images.to(device)
    labels = labels.to(device)

    images = attack(images.detach(), labels)#.cuda()

    return images