"""Train and evaluate a small convolutional neural network on MNIST."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class MnistClassifier(nn.Module):
    """A compact convolutional classifier for 28-by-28 grayscale images."""

    def __init__(self) -> None:
        """Initialize the convolutional feature extractor and classifier."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of MNIST images."""
        return self.network(images)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Examples per batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory used to download and cache MNIST.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional training-set limit for quick smoke tests.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional test-set limit for quick smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    """Select CUDA, Apple Metal, or CPU in order of availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Return at most ``max_samples`` items from a dataset."""
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("sample limits must be positive")
    return Subset(dataset, range(min(max_samples, len(dataset))))


def load_data(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Download MNIST if needed and build training and test data loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    test_dataset = limit_dataset(test_dataset, args.max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return the mean loss per example."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main() -> None:
    """Run MNIST training and print loss and accuracy after each epoch."""
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device()
    train_loader, test_loader = load_data(args)

    model = MnistClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device} with {len(train_loader.dataset):,} examples")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs}: loss={loss:.4f}, test_accuracy={accuracy:.2%}")


if __name__ == "__main__":
    main()
