import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anomalydetection.vae import CNNVAE
from src.config import DatasetConfig
from src.datagen.dataset import UnsupervisedHeatmapDataset


def init_dataloader(dataset_path: str, batch_size: int, nw: int, pin_memory: bool, transform: nn.Module):
    tensors = torch.load(dataset_path)
    dataset = UnsupervisedHeatmapDataset(**tensors, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=pin_memory,
        shuffle=True,
    )
    return loader


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def main(cfg: DatasetConfig, n_epochs: int, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = CNNVAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    loader = init_dataloader(
        cfg.get_heatmap_dataset_path(),
        batch_size=batch_size,
        nw=4 if device == 'cuda' else 0,
        pin_memory=device == 'cuda',
        transform=cfg.get_vae_transform(),
    )

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        epoch_loss = 0.0
        for hm in loader:
            hm = hm.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(hm)
            loss = vae_loss_function(recon_x, hm, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    torch.save(
        vae.state_dict(),
        cfg.get_vae_save_path(),
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on heatmap dataset")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = DatasetConfig(args.dataset)

    main(
        cfg=config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )