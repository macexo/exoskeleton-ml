# Krish's onboarding project

Trains a small convolutional neural network (two conv layers, then two fully connected layers)
to recognize handwritten digits from the MNIST dataset. It prints the training loss and test
accuracy after each of 3 epochs. MNIST downloads to `sandbox/krish/data/` on the first run.
Uses a GPU if one is available, otherwise the CPU.

## How to run

From the repo root:

```bash
uv run sandbox/krish/train.py
```
