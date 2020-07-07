from lib import *
from config import *
from edge_detector import *
import torchattacks
import copy
from torchattacks import PGD, FGSM
import time



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
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                    if (net_type in ['rgbedge', 'grayedge']) and redetect_edge: # for 
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

            if (net_type in ['rgbedge', 'grayedge']) and redetect_edge: # for 
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
    # YOU MAY NEED TO MODIFY THIS FUNCTION IN ORDER TO CHOOSE THE BEST EDGE DETECTION THAT WORKS ON YOUR DATA
    # FOR THAT, YOU MAY ALSO NEED TO CHANGE THE SOME PARAMETERS; SEE EDGE_DETECTOR.PY
    # import pdb; pdb.set_trace()

    for im in imgs:
        edge_map = edge_detect(im) 
        # edge_map = edge_map/255.
        if (edge_map.max() - edge_map.min()) > 0:
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        
        edge_map = torch.tensor(edge_map, dtype=torch.float32)
        im[-1] = edge_map # replace the last map
    
    return imgs




# def detect_edge_batch(imgs):        
#     # YOU MAY NEED TO MODIFY THIS FUNCTION IN ORDER TO CHOOSE THE BEST EDGE DETECTION THAT WORKS ON YOUR DATA
#     # FOR THAT, YOU MAY ALSO NEED TO CHANGE THE SOME PARAMETERS; SEE EDGE_DETECTOR.PY
#     # import pdb; pdb.set_trace()
#     if imgs[0].shape[-1] == 28: # hence mnist
#         for im in imgs:
#             edge_map = detect_edge_mnist(im[0][None])[0] 
#             edge_map = edge_map/255.
#             if (edge_map.max() - edge_map.min()) > 0:
#                 edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())        
#             edge_map = torch.tensor(edge_map, dtype=torch.float32)
#             im[1] = edge_map
#     else:        
#         for im in imgs:
#             # import pdb; pdb.set_trace()
#             edge_map = detect_edge_new(im[:3].permute(1,2,0)) # make it XxYx3!!! # CANNY
#             # edge_map = compute_energy_matrix(im[:3].permute(1,2,0)) # make it XxYx3!!!  # SOBEL
#             edge_map = edge_map/255.
#             edge_map = torch.tensor(edge_map, dtype=torch.float32)
            

#             im[3] = edge_map
    
#     return imgs

    
    
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()



# def train_dog_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#     use_gpu = torch.cuda.is_available()
#     best_model_wts = model.state_dict()
#     best_acc = 0.0
#     dataset_sizes = {'train': len(dataloders['train'].dataset), 
#                      'val': len(dataloders['val'].dataset)}

#     for epoch in range(num_epochs):
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train(True)
#             else:
#                 model.train(False)

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloders[phase]:
#                 if use_gpu:
#                     inputs, labels = inputs.cuda(), labels.cuda()
#                 # else:
#                 #     inputs, labels = inputs, Variable(labels)

#                 optimizer.zero_grad()

#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#                 running_loss += loss.item() #.data[0]
#     #                 import pdb; pdb.set_trace()
#                 running_corrects += torch.sum(preds == labels.data)/1.
            
#             if phase == 'train':
#                 train_epoch_loss = running_loss / dataset_sizes[phase]
#                 train_epoch_acc = running_corrects / dataset_sizes[phase]
#             else:
#                 valid_epoch_loss = running_loss / dataset_sizes[phase]
#                 valid_epoch_acc = running_corrects / dataset_sizes[phase]
                
#             if phase == 'val' and valid_epoch_acc > best_acc:
#                 best_acc = valid_epoch_acc
#                 best_model_wts = model.state_dict()

#         print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
#               'valid loss: {:.4f} acc: {:.4f}'.format(
#                 epoch, num_epochs - 1,
#                 train_epoch_loss, train_epoch_acc, 
#                 valid_epoch_loss, valid_epoch_acc))
            
#     print('Best val Acc: {:4f}'.format(best_acc))

#     model.load_state_dict(best_model_wts)
#     return model
        
        
        
        
        
