
from lib import *
from config import *
from edge_detector import *
import torchattacks
import copy
from torchattacks import PGD, FGSM

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

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                # move inputs, labels to GPU/CPU device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradients of optimizer to zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    labels = labels
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

    # device GPU or CPU?
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

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in dataloader_dict[phase]:
                # move inputs, labels to GPU/CPU device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # borji; perform the attack with probability .5
#                 if torch.rand(1).item() < .5:
                # import pdb; pdb.set_trace()

#                     inputs = inputs.clone(requires_grad = True)
#                 inputs = (inputs-inputs.min()) / (inputs.max()-inputs.min())    
                
                inputs_adv = attack(inputs, labels)#.cuda()
                
                # set gradients of optimizer to zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, axis=1)


                    # with newly computed edge map  
                    if (net_type.lower() not in ['rgb', 'gray', 'edge']) and redetect_edge: # for 
                        # import pdb; pdb.set_trace()
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

#     import pdb; pdb.set_trace()

    torch.save(net.state_dict(), save_path)
#     return net




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

            if (net_type not in ['rgb', 'gray', 'edge']) and redetect_edge: # or it has 4 channels?
              images = detect_edge_batch(images)
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

    # train on gpu, load model on cpu machine
    # load_weights = torch.load(model_path, map_location=("cuda:0", "cpu"))
    # net.load_state_dict(load_weights)
    
    # display fine-tuning model's architecture
#     for name, param in net.named_parameters():
#         print(name, param)
#     return net
        
        

# def detect_edge(img):        
#     # borji
#     edge_map = detect_edge_new(img.permute(1,2,0)) # make it [x,y,3]!!!
#     edge_map = edge_map/255.
#     edge_map = torch.tensor(edge_map, dtype=torch.float32)
    
#     return edge_map
    
    
def detect_edge_batch(imgs):        
    # borji
#     import pdb; pdb.set_trace()
    if imgs[0].shape[-1] == 28: # hence mnist
        for im in imgs:
            edge_map = detect_edge_mnist(im[0][None])[0] 
            edge_map = edge_map/255.
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            im[1] = edge_map
    else:        
        for im in imgs:
            edge_map = detect_edge_new(im[:3].permute(1,2,0)) # make it XxYx3!!!
            edge_map = edge_map/255.
            edge_map = torch.tensor(edge_map, dtype=torch.float32)
            im[3] = edge_map
    
    return imgs

    
    
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
    
    
    
