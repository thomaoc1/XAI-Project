import argparse

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.classification.binary_classifier import BinaryClassifier
from src.config import DatasetConfig


def init_dataloader(dataset_path: str, batch_size: int, nw: int, transform, pin_memory: bool):
    dataset = ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        shuffle=True,
        pin_memory=pin_memory
    )
    return loader


def main(cfg: DatasetConfig, num_epochs: int, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    loader = init_dataloader(
        cfg.get_classifier_dataset_split('train'),
        batch_size,
        num_workers,
        pin_memory=device == 'cuda',
        transform=cfg.get_classifier_transform()
    )

    model = BinaryClassifier().to(device)
    optimiser = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for img, label in progress_bar:
            img, label = img.to(device), label.to(device)

            preds = model(img).squeeze(1)
            loss = F.cross_entropy(preds, label)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), cfg.get_classifier_save_path())

def parse_args():
    parser = argparse.ArgumentParser(description="Training Classifier")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = DatasetConfig(args.dataset)
    main(
        cfg=config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
