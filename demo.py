import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from src.conv_numpy import conv2d_numpy

# load image
img = Image.open("tesla_fsd.jpg").convert("RGB")
img_np = np.array(img).astype(np.float32) / 255.0
x_np = img_np.transpose(2, 0, 1)[None, ...] # (1, 3, H, W)

# edge filter
w = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
w = np.stack([w, w, w])[None, ...] # (1, 3, 3, 3)

# numpy conv2d
out_np = conv2d_numpy(x_np, w, padding=1)
edge_map_np = np.abs(out_np[0, 0])

# pytorch conv2d (1 line)
x_torch = torch.from_numpy(x_np)
w_torch = torch.from_numpy(w)
out_torch = F.conv2d(x_torch, w_torch, padding=1)
edge_map_torch = out_torch[0, 0].numpy().copy()

# plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Tesla FSD image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Your NumPy conv -> edge map")
plt.imshow(edge_map_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Pytorch Conv -> Edge map")
plt.imshow(edge_map_torch, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()