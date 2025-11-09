# ResNet-18 from Scratch → CIFAR-10 → FastAPI Deploy

 Sprint to build **ResNet-18 from scratch (with AI (Grok) debugging/navigation and as a 'senior engineer' co pilot through the project) and deploy it as a **FastAPI service**.

I am a CS student learning how to build real ML systems with the help of AI from scratch.
---

## Day 1: NumPy Conv2d from Scratch
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