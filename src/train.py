import os
from typing import Optional

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from src.classifier import DeepFakeClassifier

def init_dataloader(batch_size, dev, nw):
    dataset = ImageFolder("dataset/deepfake-dataset/train", transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        shuffle=True,
        pin_memory=dev == 'cuda'
    )
    return loader


def train(num_epochs: int, batch_size: int, nw: int, dev: str):
    loader = init_dataloader(batch_size, dev, nw)

    model = DeepFakeClassifier().to(dev)
    optimiser = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for img, label in progress_bar:
            img, label = img.to(dev), F.one_hot(label.long(), 2).float().to(dev)

            preds = model(img).squeeze(1)
            loss = F.cross_entropy(preds, label)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'model.pt')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0
    train(
        num_epochs=2,
        batch_size=64,
        nw=num_workers,
        dev=device,
    )