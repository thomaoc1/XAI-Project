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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = CNNVAE(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    num_epochs = 50

    loader = init_dataloader(
        'deep_fake_hm_dataset.pt',
        batch_size=64,
        pin_memory=device == 'cuda',
        nw=4 if device == 'cuda' else 0
    )

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0.0
        for hm in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
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
        'cnn_vae_model.pt'
    )


if __name__ == '__main__':
    main()