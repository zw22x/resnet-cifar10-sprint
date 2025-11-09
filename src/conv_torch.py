import torch 
import torch.nn.functional as F
from typing import Tuple

def conv2d_torch(x: torch.Tensor, # (N, C, H, W)
                 w: torch.Tensor, # (F, C, KH, KW)
                 stride: int = 1,
                 padding: int = 0) -> torch.Tensor:
        """ PyTorch 2D convolution - 1 line. Matches NumPy version exactly."""
        return F.conv2d(x, w, stride=stride, padding=padding)

