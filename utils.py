from lib import *
from config import *
from edge_detector import *
import torchattacks
# import copy
from torchattacks import PGD, FGSM
import time
import numpy as np
from PIL import Image



def make_datapath_list(phase='train'):
    rootpath = './dataset/'
    target_path = osp.join(rootpath + phase + "/**/*.jpg")

    path_list = [path for path in glob.glob(target_path)]
    return path_list

# Train model
def train_model(net, dataloader_dict, criterior, optimizer, num_epochs, save_path):

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)

    # boost network speed on gpu
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        # for phase in ['train', 'val']:
        for phase in ['train']:            
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # if (epoch == 0) and (phase == 'train'):
            #     continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                # import pdb; pdb.set_trace()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    # labels = labels
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, axis=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.shape[0]
                    epoch_corrects += torch.sum(preds==labels.data)


            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict(), save_path)



# Train model
def train_robust_model(net, dataloader_dict, criterior, optimizer, num_epochs, save_path, attack_type = 'FGSM', eps=8/255, net_type='rgb', redetect_edge=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)


    # make a copy of the net first
    # net_copy = copy.deepcopy(net)
    # fgsm_attack = FGSM(net_copy, eps=eps)    

    if attack_type.upper() == 'FGSM':
        attack = FGSM(net, eps=eps)    
    else:    
        attack = PGD(net, eps=eps, alpha=8/255, iters=40)    


    # boost network speed on gpu
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        # for phase in ['train', 'val']:
        for phase in ['train']:            
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            epoch_corrects_adv = 0

            # if (epoch == 0) and (phase == 'train'):
            #     continue

            for inputs, labels in dataloader_dict[phase]:
                # import pdb; pdb.set_trace()

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs_adv = attack(inputs, labels)#.cuda()
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, axis=1)


                    # with newly computed edge map  
                    # if (net_type not in ['rgb', 'gray', 'edge']) and redetect_edge: # for 
                    if (net_type in ['rgbedge', 'greyedge']) and redetect_edge: # for 
                        inputs_adv = detect_edge_batch(inputs_adv);
                        # pass

                    outputs_adv = net(inputs_adv)
                    loss_adv = criterior(outputs_adv, labels)
                    _, preds_adv = torch.max(outputs_adv, axis=1)

                    alpha = .5
                    loss = alpha*loss + (1-alpha)*loss_adv


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.shape[0]
                    epoch_corrects += torch.sum(preds==labels.data)
                    epoch_corrects_adv += torch.sum(preds_adv==labels.data)



            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            epoch_accuracy_adv = epoch_corrects_adv.double() / len(dataloader_dict[phase].dataset)


            print("{} Loss: {:.4f}, Acc: {:.4f}, Acc_adv: {:.4f}".format(phase, epoch_loss, epoch_accuracy, epoch_accuracy_adv))

    torch.save(net.state_dict(), save_path)



def test_model_clean(net, dataloader_dict):
    correct = 0
    total = 0

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    net.eval()
    
    
    for images, labels in dataloader_dict['val']:    
        # images = (images-images.min()) / (images.max()-images.min())
        images, labels = images.to(device), labels.to(device)         
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
    #     correct += (predicted == labels.cuda()).sum()
        correct += (predicted == labels).sum()

    acc = float(correct) / total
        
    return acc, images    


def test_model_attack(net, dataloader_dict, epsilons, attack_type = 'FGSM', net_type='rgb', redetect_edge=False):
    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)

    net.eval()
    accuracies = []; examples = []

    for eps in epsilons:

        if attack_type.upper() == 'FGSM':
            attack = FGSM(net, eps=eps)    
        else:    
            attack = PGD(net, eps=eps, alpha=8/255, iters=40)      
        
            
        correct = 0; total = 0;       

        # import pdb; pdb.set_trace()
        for images, labels in dataloader_dict['val']:
            images, labels = images.to(device), labels.to(device)         
            # images = (images-images.min()) / (images.max()-images.min())                
            images = attack(images, labels)#.cuda()

            outputs = net(images)

            if (net_type in ['rgbedge', 'grayedge']) and redetect_edge: # for 
              # import pdb;pdb.set_trace()
              images = detect_edge_batch(images)
              outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        acc = float(correct) / total
        accuracies.append(acc)

    return accuracies, images    








# Train model
def train_substitue_model(net, substitute_net, dataloader_dict, optimizer, num_epochs, save_path):

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    substitute_net.to(device)

    # boost network speed on gpu
    torch.backends.cudnn.benchmark = True


    substitute_net.train()
    net.eval()

    phase = 'train'

    loss_fn = nn.BCELoss()    
    softmax = nn.Softmax()

    # optimizer = optim.Adadelta(substitute_net.parameters())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        epoch_loss = 0.0
        epoch_corrects = 0

        for inputs, labels in tqdm(dataloader_dict[phase]):
            # import pdb; pdb.set_trace()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs_sub = substitute_net(inputs[:,:-1]) # outputs_sub is the logits now; substitute model is only trained on img channels not edge map
                outputs = net(inputs)

                # loss = criterior(outputs, labels)
                # import pdb; pdb.set_trace()
                loss = loss_fn(softmax(outputs_sub), softmax(outputs.detach()))        
                _, preds = torch.max(outputs_sub, axis=1)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.shape[0]
                epoch_corrects += torch.sum(preds==labels.data)


        epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
        epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

        print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(substitute_net.state_dict(), save_path)




def test_model_BPDA_attack(net, substitute_net, dataloader_dict, epsilons, attack_type = 'FGSM', redetect_edge=True):
    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    substitute_net.to(device)

    net.eval()
    substitute_net.eval()

    accuracies = []; examples = []

    for eps in epsilons:

        if attack_type.upper() == 'FGSM':
            attack = FGSM(substitute_net, eps=eps)    
        else:    
            attack = PGD(substitute_net, eps=eps, alpha=8/255, iters=40)      
        
            
        correct = 0; total = 0;       

        # import pdb; pdb.set_trace()
        for images, labels in dataloader_dict['val']:
            images, labels = images.to(device), labels.to(device)         
            # images = (images-images.min()) / (images.max()-images.min())                
            images_attacked = attack(images[:,:-1], labels)#.cuda() # exclude the edge map; attack Image part only
            # images = attack(images, labels)#.cuda() # exclude the edge map

            # outputs = net(images)

            # if (net_type in ['rgbedge', 'grayedge']) and redetect_edge: # BPDA is only for x+edge models
              # import pdb;pdb.set_trace()
            import pdb; pdb.set_trace()
            # edge_maps = torch.zeros((images.shape[0],1,images.shape[2],images.shape[2]))              
            # data = torch.cat((images_attacked, edge_maps),dim=1)#[None]
            images[:,:-1] = images_attacked # replace the healthy image with attacked one
            if redetect_edge:
                images = detect_edge_batch(images) # compute the edge from the image and append it back; make sure EDGE_ALL_CHANNELS in config.py is False!!!
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        acc = float(correct) / total
        accuracies.append(acc)

    return accuracies, images    









def load_model(net, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    load_weights = torch.load(model_path, map_location=device)
    net.load_state_dict(load_weights)



def detect_edge_batch(imgs):        
    # YOU MAY NEED TO MODIFY THIS FUNCTION IN ORDER TO CHOOSE THE BEST EDGE DETECTION THAT WORKS ON YOUR DATA
    # FOR THAT, YOU MAY ALSO NEED TO CHANGE THE SOME PARAMETERS; SEE EDGE_DETECTOR.PY
    # import pdb; pdb.set_trace()

    for im in imgs:
        if EDGE_ALL_CHANNELS:
            edge_map = edge_detect(im) # discard the last channel since it already has an edge map!! which we want to replace        
        else:    
            edge_map = edge_detect(im[:-1]) # discard the last channel since it already has an edge map!! which we want to replace

        # import pdb; pdb.set_trace()
        # edge_map = edge_map/255.
        if (edge_map.max() - edge_map.min()) > 0:
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        
        edge_map = torch.tensor(edge_map, dtype=torch.float32)
        im[-1] = edge_map # replace the last map
    
    return imgs

    
    
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
