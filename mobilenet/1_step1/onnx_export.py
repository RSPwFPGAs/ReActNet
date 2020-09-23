import torch
import torch.onnx

from reactnet import reactnet
 
m = reactnet()
 
d = torch.rand(1, 3, 416, 416)
o = m(d)
 
onnx_path = "onnx_reactnet.1_step1.onnx"
torch.onnx.export(m, d, onnx_path)
 
#netron.start(onnx_path)
