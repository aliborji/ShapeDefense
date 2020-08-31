

import torch


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

epsilon = (8 / 255.) / std
alpha = (10 / 255.) / std
pgd_alpha = (2 / 255.) / std



# RGB
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


#if net_type == 'rgbedge' then uncomment the following lines
#upper_limit = torch.cat((upper_limit, upper_limit.mean().reshape(1,1,1)))
#lower_limit = torch.cat((lower_limit, lower_limit.mean().reshape(1,1,1)))
#epsilon = torch.cat((epsilon, epsilon.mean().reshape(1,1,1)))
#alpha = torch.cat((alpha, alpha.mean().reshape(1,1,1)))
#pgd_alpha = torch.cat((pgd_alpha, pgd_alpha.mean().reshape(1,1,1)))



#if net_type == 'edge':
#upper_limit = upper_limit.mean()
#lower_limit = lower_limit.mean()
#epsilon = epsilon.mean().reshape(1,1,1)
#alpha = alpha.mean()
#pgd_alpha = pgd_alpha.mean()




