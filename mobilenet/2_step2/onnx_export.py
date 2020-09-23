import torch
import torch.onnx

from reactnet import reactnet
 
m = reactnet()
 
d = torch.rand(1, 3, 416, 416)
o = m(d)
 
onnx_path = "onnx_reactnet.2_step2.onnx"
torch.onnx.export(m, d, onnx_path)
 
#netron.start(onnx_path)
