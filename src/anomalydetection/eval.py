import torch
import os
import torchattacks
from tqdm import tqdm

from pytorch_grad_cam import GradCAM

from src.anomalydetection.vae import CNNVAE
from src.classifier.binary_classifier import BinaryClassifier
from src.classifier.eval import init_dataloader


def compute_elbo(x, recon, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction='none').flatten(1).mean(1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon_loss + kl

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load('new_model.pt', map_location=device, weights_only=True))
    model.eval()

    vae_model = CNNVAE(latent_dim=128).to(device)
    vae_model.load_state_dict(torch.load('cnn_vae_model.pt', map_location=device, weights_only=True))
    vae_model.eval()

    attack = torchattacks.FGSM(model)

    target_layers = [model.backbone.layer4[-1]]

    loader = init_dataloader(batch_size=64, dev=device, nw=num_workers)

    all_score_clean = []
    all_score_adv = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for img, label in tqdm(loader, desc='Processing'):
            img, label = img.to(device), label.to(device)

            grayscale_cam = torch.tensor(cam(input_tensor=img)).unsqueeze(1).to(device)

            adv_img = attack(img, label)
            grayscale_cam_adv = torch.tensor(cam(input_tensor=adv_img)).unsqueeze(1).to(device)

            recon, mu, logvar = vae_model(grayscale_cam)
            recon_adv, mu_adv, logvar_adv = vae_model(grayscale_cam_adv)

            score_clean = compute_elbo(grayscale_cam, recon, mu, logvar)
            score_adv = compute_elbo(grayscale_cam_adv, recon_adv, mu_adv, logvar_adv)

            all_score_clean.append(score_clean.detach().cpu())
            all_score_adv.append(score_adv.detach().cpu())

    all_score_clean = torch.cat(all_score_clean)
    all_score_adv = torch.cat(all_score_adv)

    print(f'Clean Mean: {all_score_clean.mean().item():.4f}, Std: {all_score_clean.std().item():.4f}')
    print(f'Adv   Mean: {all_score_adv.mean().item():.4f}, Std: {all_score_adv.std().item():.4f}')

    torch.save({'clean': all_score_clean, 'adv': all_score_adv}, 'vae_elbo_scores.pt')


if __name__ == '__main__':
    main()
