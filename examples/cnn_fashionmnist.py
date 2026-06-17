"""CNN training on FashionMNIST — compatible with run_custom_training() on OpenShift.

Pass the contents of this file as the `script` parameter to run_custom_training().
Writes data to /tmp/data and checkpoint to /tmp/output (both covered by the
required OpenShift emptyDir volumes for /tmp).
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def train():
    if "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Device: {device}  world_size={os.environ.get('WORLD_SIZE', '1')}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.FashionMNIST(
        root="/tmp/data", train=True, download=True, transform=transform
    )
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    loader = DataLoader(
        dataset, batch_size=64, sampler=sampler, shuffle=(sampler is None), num_workers=2
    )

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = CNN().to(device)
    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = int(os.environ.get("EPOCHS", "3"))

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        if rank == 0:
            acc = 100.0 * correct / total
            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"loss={total_loss / len(loader):.4f}  acc={acc:.2f}%",
                flush=True,
            )

    if rank == 0:
        os.makedirs("/tmp/output", exist_ok=True)
        torch.save(model.state_dict(), "/tmp/output/cnn_fashionmnist.pt")
        print("Checkpoint saved → /tmp/output/cnn_fashionmnist.pt", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


train()
