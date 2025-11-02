# ResNet-18 from Scratch → CIFAR-10 → FastAPI Deploy

7-day sprint to build **ResNet-18 from scratch** and deploy it as a **FastAPI service**.

---

## Day 1: NumPy Conv2d from Scratch
- Implemented **2D convolution** in **pure NumPy**
- Matches **PyTorch `F.conv2d`** within `1e-5` (floating-point tolerance)
- Full test suite with `pytest`
- `poetry` + `pytest.ini` + `src/` layout

```bash
poetry run pytest -q
# 1 passed — SUCCESS!
