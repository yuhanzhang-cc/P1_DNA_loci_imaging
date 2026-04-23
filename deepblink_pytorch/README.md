# deepblink_pytorch

PyTorch migration of deepBlink core functionality.

## Quick start

```bash
pip install -e .
```

## CLI

```bash
deepblink-torch create --images ./images --labels ./labels --output ./dataset.npz
deepblink-torch train --data ./dataset.npz --epochs 50 --batch-size 16 --lr 1e-3 --device cuda
deepblink-torch predict --model ./artifacts/checkpoints/best_model.pt --images ./images --output ./predictions --probability 0.5 --device cuda
```

You can also run training directly:

```bash
python train.py --data ./dataset.npz --epochs 50 --batch-size 16 --lr 1e-3 --device cuda
```

