import time
import torch
import numpy as np
from src.conv_numpy import conv2d_numpy
from src.conv_torch import conv2d_torch
import wandb

# Config
N, C, H, W = 1, 3, 128, 128 # 1 image, 3 channels, 512x512
F, KH, KW = 64, 3, 3 # 64 filters, 3x3 kernel
stride, padding = 1, 1
device = 'cpu' # change to cuda if GPU is available or later

# Data
torch.manual_seed(42)
np.random.seed(42)

x_torch = torch.randn(N, C, H, W, device=device)
w_torch = torch.randn(F, C, KH, KW, device=device)
x_np = x_torch.cpu().numpy()
w_np = w_torch.cpu().numpy()

# Warm Up
for _ in range(5):
    conv2d_numpy(x_np, w_np, stride, padding)
    conv2d_torch(x_torch, w_torch, stride, padding)

#Benchmark 
def time_func(func, *args):
    start = time.time()
    for _ in range(10):
        func(*args)
    return (time.time() - start) / 10
time_numpy = time_func(conv2d_numpy, x_np, w_np, stride, padding)
time_torch = time_func(conv2d_torch, x_torch, w_torch, stride, padding)
print(f"Numpy: {time_numpy: .5f}s")
print(f"Torch: ({device}): {time_torch: .5f}s")
print(f"Speedup: {time_numpy / time_torch: 1f}x")

# W&B Log
wandb.init(project="resnet-sprint", name="day2-benchmark")
wandb.log({"input_size": f"{H}x{W}", 
           "filters": F, 
           "time_numpy": time_numpy, 
           "time_torch": time_torch, 
           "speedup": time_numpy / time_torch, 
           "device": device})
wandb.finish()
