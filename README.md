# ResNet-18 from Scratch → CIFAR-10 → FastAPI Deploy

 Sprint to build **ResNet-18 from scratch (with AI (Grok) debugging/navigation and as a 'senior engineer' co pilot through the project) and deploy it as a **FastAPI service**.

I am a CS student learning how to build real ML systems with the help of AI from scratch.
---

 NumPy Conv2d from Scratch
- Implemented **2D convolution** in **pure NumPy**
- Matches **PyTorch `F.conv2d`** within `1e-5` (floating-point tolerance)
- Full test suite with `pytest`
- `poetry` + `pytest.ini` + `src/` layout


```bash
poetry run pytest -q
# 1 passed — SUCCESS!

### Benchmark: NumPy vs PyTorch (CPU)
- **Input**: 128×128 RGB image, 16 filters (3×3), stride=1, padding=1
- **NumPy (from scratch)**: Pure Python loops → **2.08 seconds**
- **PyTorch (`F.conv2d`)**: Optimized C++/CPU backend → **0.00017 seconds**
- **Result**: **12,493× speedup** with just one line of code
- **Takeaway**: You now understand *why* deep learning frameworks exist.

```bash
poetry run python benchmark.py
# Output:
# NumPy:  2.08393s
# PyTorch (cpu): 0.00017s
# Speedup: 12493.788308x

---

## Tesla FSD Vision Demo — Real Camera Image
- Used Tesla FSD camera image (`tesla_fsd.jpg`)URL: https://www.google.com/imgres?q=tesla%20road%20image%20from%20front%20camera&imgurl=https%3A%2F%2Fteslamotorsclub.com%2Ftmc%2Fattachments%2Ffront-camera-model-s-lr-png.598369%2F%3Fid%3D5064820&imgrefurl=https%3A%2F%2Fteslamotorsclub.com%2Ftmc%2Fthreads%2Fvideo-issue-from-front-camera-on-new-lr.209692%2F&docid=GAi6_CXV631tXM&tbnid=fNHtZI9qb2hwtM&vet=12ahUKEwjOnbmnpeSQAxWvGtAFHQODOisQM3oECB4QAA..i&w=2000&h=1498&hcb=2&ved=2ahUKEwjOnbmnpeSQAxWvGtAFHQODOisQM3oECB4QAA
- Ran **my `conv2d_numpy`** (from scratch) → **detected lane lines, cars, trees**
- Ran **PyTorch `F.conv2d`** → **pixel-perfect match**
- **Result**: My code **sees like Tesla** — **Layer 1 of FSD vision

Output: 3-panel plot

Raw FSD image
My NumPy conv → glowing edges
PyTorch conv → identical

```bash
poetry run python demo.py

---

Residual Block — ResNet from Scratch
- Built **ResNet's core magic**: `x + F(x)` with **skip connections**
- `ResidualBlock` in pure PyTorch — **no `torchvision`**
- Handles **stride changes** and **channel mismatches**
- Verified: `32×32×3 → 32×32×64` output

```bash
poetry run python src/residual.py
# Input:  torch.Size([1, 3, 32, 32])
# Output: torch.Size([1, 64, 32, 32])

---

Full ResNet-18 from Scratch (No `torchvision`)
- Built **complete ResNet-18** in pure PyTorch
- 18 layers: 4 stages (64→128→256→512 channels)
- Skip connections** + BatchNorm** + **ReLU**
- Fixed **BatchNorm batch size 1 error** → test with batch size 2
- Verified: `(2, 3, 32, 32) → (2, 10)` output

```bash
poetry run python src/resnet.py
# Input:  torch.Size([2, 3, 32, 32])
# Output: torch.Size([2, 10])

Used Grok as a senior engineer co-pilot for:
Debugging Poetry, NumPy 2, PyTorch, MacOS issues
Optimizing benchmark from 3 min → 2 sec
Explaining convolution with Tesla FSD visuals
Writing clean, production-grade code

I typed every line. I ran every test. I own the result.

Stack

Python 3.12
Poetry (dependency + env management)
PyTorch (reference + GPU-ready)
NumPy (from scratch)
pytest (testing)
GitHub (CI/CD ready)
