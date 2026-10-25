import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="sandbox/manasvi/data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Training finished. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
