import numpy as np
from typing import Tuple

def conv2d_numpy(x: np.ndarray, # (N, C, H, W)
                 w: np.ndarray, # (F, C, KH, KW)
                 stride: int = 1,
                 padding: int = 0) -> np.ndarray:
    """ pure numpy 2d convolution (no bias). matches torch.nn.functional.conv2d exactly. """
    
    
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)))

    N, C, H, W = x.shape
    F, _, KH, KW = w.shape 
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1

    out = np.zeros((N, F, OH, OW), dtype=x.dtype)

    for n in range(N):
        for f in range(F):
            for oh in range(OH):
                for ow in range(OW):
                     h_start = oh * stride
                     w_start = ow * stride
                     window = x[n, :, h_start:h_start+KH, w_start:w_start+KW]
                     out[n, f, oh, ow] = np.sum(window * w[f])

    return out