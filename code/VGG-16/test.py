import torch
print(torch.__version__)  # PyTorch 版本
print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
#import pycuda.driver as cuda
#import pycuda.autoinit

#(major, minor) = cuda.Device(0).compute_capability()
#print(f"Compute Capability: {major}.{minor}")
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
full_precision_tensor = torch.rand(5)
print("Full Precision: {}".format(full_precision_tensor))
low_precision_tensor = float_quantize(full_precision_tensor, exp=5, man=2, rounding="nearest")
print("Low Precision: {}".format(low_precision_tensor))
