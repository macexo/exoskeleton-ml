"""Train a small convolutional network to classify MNIST digits."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3


def build_model() -> nn.Module:
    """Build a small CNN for 28x28 grayscale digit images.

    Returns:
        A model mapping a batch of images (N, 1, 28, 28) to class logits (N, 10).
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one pass over the training data.

    Args:
        model: The network to train.
        loader: Training data loader.
        optimizer: Optimizer updating the model's parameters.
        loss_fn: Loss function comparing logits to labels.
        device: Device to run on.

    Returns:
        The average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on a dataset.

    Args:
        model: The network to evaluate.
        loader: Data loader for the evaluation set.
        device: Device to run on.

    Returns:
        The fraction of correctly classified examples.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    """Download MNIST, train the CNN, and print test accuracy after each epoch."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device}")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, test accuracy {accuracy:.2%}")


if __name__ == "__main__":
    main()
