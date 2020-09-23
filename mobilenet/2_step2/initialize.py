import torch
import torch.onnx

from reactnet import reactnet


# create ReActNet model
model = reactnet()
#for key, v in enumerate(model.state_dict()):
#    print(key, v)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    if param_tensor.find('binary') != -1 :
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print()

# original saved file with DataParallel
pretrained_net = torch.load('checkpoint.pth.tar', map_location=torch.device('cpu'))
#for key, v in enumerate(pretrained_net):
#    print(key, v)
#print("----\n epoch: ", pretrained_net['epoch'])
#print("----\n state_dict: ", pretrained_net['state_dict'])
#print("----\n best_top1_acc: ", pretrained_net['best_top1_acc'])
#print("----\n optimizer: ", pretrained_net['optimizer'])

print("Checkpoint's state_dict:")
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in pretrained_net['state_dict'].items():
    name = k[7:] # remove `module.`
    if name.endswith('weights') :
        #name = name[:-1] # rename feature.xx.yy.weights with feature.xx.yy.weight
        print(name, "\t", v.size())
    new_state_dict[name] = v
print()
# load params
model.load_state_dict(new_state_dict)


dummy_input = torch.rand(1, 3, 416, 416)
onnx_path = "onnx_reactnet.trained.onnx"
torch.onnx.export(model, dummy_input, onnx_path)
 


