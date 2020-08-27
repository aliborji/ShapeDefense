import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, net_type='rgb'):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        if net_type == 'rgb':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        elif net_type == 'edge':    
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:    
            self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(net_type='rgb'):
    return PreActResNet(PreActBlock, [2,2,2,2], 10, net_type)



def define_model(net_type):

    NUM_CLASSES = 10
    resnet = models.resnet18(pretrained=True)

    # import pdb; pdb.set_trace()
    if net_type.lower() == 'rgb':
          # new_layer = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
          # resnet.conv1 = new_layer
        pass  


        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
    elif net_type.lower() == 'edge': # rgb_egde
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

    return resnet

