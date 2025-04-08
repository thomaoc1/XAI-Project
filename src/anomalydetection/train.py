import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anomalydetection.vae import CNNVAE
from src.datagen.dataset import UnsupervisedHeatmapDataset


def init_dataloader(dataset_path: str, batch_size: int, nw: int, pin_memory: bool):
    tensors = torch.load(dataset_path)
    dataset = UnsupervisedHeatmapDataset(**tensors)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=pin_memory,
        shuffle=True,
    )
    return loader


def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def get_paths(dataset_name: str, attack_name: str):
    dataset_path = os.path.join('dataset', 'heatmap', f'{dataset_name}_{attack_name.lower()}_hm_dataset.pt')
    save_path = os.path.join('model', f'{dataset_name}_{attack_name.lower()}_hm_vae.pt')
    return dataset_path, save_path


def main(dataset_name: str, attack_name: str, n_epochs: int, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = CNNVAE(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    dataset_path, save_path = get_paths(dataset_name, attack_name)

    loader = init_dataloader(
        dataset_path,
        batch_size=batch_size,
        pin_memory=device == 'cuda',
        nw=4 if device == 'cuda' else 0
    )

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        epoch_loss = 0.0
        for hm in loader:
            hm = hm.to(device).unsqueeze(1)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(hm)
            loss = loss_function(recon_x, hm, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    torch.save(
        vae.state_dict(),
        save_path,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on heatmap dataset")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('attack', type=str, choices=["FGSM", "PGD"])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        dataset_name=args.dataset,
        attack_name=args.attack,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )