import numpy as np
import torch
import torch.nn.functional as F
from src.conv_numpy import conv2d_numpy

def test_conv_matches_torch():
    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(1, 3, 32, 32).numpy()
    w = torch.randn(16, 3, 3, 3).numpy()

    y_np = conv2d_numpy(x, w, stride=1, padding=1)
    y_torch = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), stride=1, padding=1).numpy()

    assert y_np.shape == y_torch.shape, f"Shape mismatch: {y_np.shape} vs {y_torch.shape}"
    assert np.allclose(y_np, y_torch, atol=1e-5), \
        f"Max diff: {np.abs(y_np - y_torch).max():.10f}"

    print("SUCCESS: Your NumPy conv matches PyTorch exactly!")