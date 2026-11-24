"""Train a small CNN on MNIST.

Run with: uv run sandbox/krishiv/train.py
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parent / "data"


class SmallCNN(nn.Module):
    """Two-layer convolutional network for 28x28 grayscale digit images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Batch of images with shape (batch, 1, 28, 28).

        Returns:
            Logits with shape (batch, 10).
        """
        return self.classifier(self.features(x))


def get_device() -> torch.device:
    """Pick the best available device (CUDA, then MPS, then CPU).

    Returns:
        The torch device to train on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Build MNIST train and test data loaders, downloading the data if needed.

    Args:
        batch_size: Number of images per training batch.

    Returns:
        A (train_loader, test_loader) tuple.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=1000),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over the training data.

    Args:
        model: The model to train.
        loader: Training data loader.
        optimizer: Optimizer updating the model's parameters.
        device: Device the model and batches live on.

    Returns:
        Mean cross-entropy loss over the epoch.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy.

    Args:
        model: The model to evaluate.
        loader: Data loader to evaluate on.
        device: Device the model and batches live on.

    Returns:
        Fraction of images classified correctly.
    """
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    """Parse arguments, train the model, and print per-epoch metrics."""
    parser = argparse.ArgumentParser(description="Train a small CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(args.batch_size)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
