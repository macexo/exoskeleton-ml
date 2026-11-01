import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(
    "sandbox/saharsh/data",
    train=True,
    download=True,
    transform=transform,
)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for images, labels in train_loader:
    predictions = model(images)
    loss = loss_fn(predictions, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training complete. Final batch loss: {loss.item():.4f}")
