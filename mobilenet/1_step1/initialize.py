import torch
import torch.nn as nn
import torch.nn.functional as F

from reactnet import reactnet

 
m = reactnet()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in m.state_dict():
    print(param_tensor, "\t", m.state_dict()[param_tensor].size())


pretrained_net = torch.load('checkpoint.pth.tar', map_location=torch.device('cpu'))
print("----\nepoch: ", pretrained_net['epoch'])
print("----\nstate_dict: ", pretrained_net['state_dict'])
print("----\nbest_top1_acc: ", pretrained_net['best_top1_acc'])
print("----\noptimizer: ", pretrained_net['optimizer'])


#for key, v in enumerate(pretrained_net):
#    print(key, v)


