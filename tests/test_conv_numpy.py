import numpy as np
from src.conv_numpy import conv2d_numpy

def test_conv2d_numpy():
    # Small random input
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)
    w = np.random.randn(1, 1, 3, 3).astype(np.float32)
    
    # Run your NumPy conv
    out_numpy = conv2d_numpy(x, w, stride=1, padding=1)
    
    # Run PyTorch (ground truth)
    import torch
    x_torch = torch.from_numpy(x)
    w_torch = torch.from_numpy(w)
    out_torch = torch.nn.functional.conv2d(x_torch, w_torch, stride=1, padding=1)
    
    # Compare within 1e-5
    np.testing.assert_allclose(out_numpy, out_torch.numpy(), atol=1e-5)
    print("Your conv matches PyTorch exactly!")
