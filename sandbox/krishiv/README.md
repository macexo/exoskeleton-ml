# krishiv onboarding project

A small CNN trained on MNIST with PyTorch and torchvision.

## What it does

`train.py` downloads MNIST into `sandbox/krishiv/data/` (ignored by Git), trains a two-layer convolutional network with Adam, and prints the training loss and test accuracy after each epoch. It uses CUDA or MPS if available, otherwise CPU.

## How to run

From the repo root:

```
uv run sandbox/krishiv/train.py
```

Optional flags:

```
uv run sandbox/krishiv/train.py --epochs 5 --batch-size 128 --lr 0.001 --seed 0
```

Defaults are 3 epochs, batch size 64, learning rate 1e-3, seed 0. Expect test accuracy around 98% after 3 epochs.
