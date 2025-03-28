import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np

from src.classifier import DeepFakeClassifier

def init_dataloader(batch_size, dev, nw):
    dataset = ImageFolder("dataset/deepfake-dataset/train", transform=transforms.ToTensor())

    dominant_class = 0
    min_class = 1
    downsampling_factor = 3

    targets = np.array(dataset.targets)
    dominant_indices = np.where(targets == dominant_class)[0]
    min_class_indices = np.where(targets == min_class)[0]

    downsampled_dominant_indices = np.random.choice(dominant_indices, len(dominant_indices) // downsampling_factor, replace=False)
    final_indices = np.concatenate([downsampled_dominant_indices, min_class_indices])

    sampler = SubsetRandomSampler(final_indices)


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=dev == 'cuda'
    )

    return loader, downsampling_factor


def train(num_epochs: int, batch_size: int, nw: int, dev: str):
    loader, downsample_factor = init_dataloader(batch_size, dev, nw)

    model = DeepFakeClassifier().to(dev)
    optimiser = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for img, label in progress_bar:
            img, label = img.to(dev), label.float().to(dev)

            weight = torch.ones_like(label, device=dev)
            weight[label == 0] = downsample_factor

            preds = model(img).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(preds, label, weight=weight)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'imbalance-model.pt')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0
    train(
        num_epochs=2,
        batch_size=64,
        nw=num_workers,
        dev=device,
    )