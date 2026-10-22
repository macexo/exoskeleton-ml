# Yug's MNIST Classifier

This onboarding project trains a small convolutional neural network to recognize handwritten
digits from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. The script uses
PyTorch and torchvision, automatically downloads the dataset into the ignored `data/` directory,
and reports test accuracy after each epoch.

## Run it

From the repository root, install the project dependencies and start training:

```bash
uv sync
uv run sandbox/yug/train.py
```

The default run trains for three epochs. To change the number of epochs or batch size:

```bash
uv run sandbox/yug/train.py --epochs 5 --batch-size 128
```

For a quick end-to-end smoke test, limit the number of examples:

```bash
uv run sandbox/yug/train.py --epochs 1 --max-train-samples 512 --max-test-samples 256
```

Run `uv run sandbox/yug/train.py --help` to see all available options. CUDA and Apple Metal are
used automatically when available; otherwise, the model trains on the CPU.
