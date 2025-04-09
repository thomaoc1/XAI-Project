import argparse

import torch
import torchattacks
from pytorch_grad_cam import GradCAM
from tqdm import tqdm

from src.anomalydetection.eval_vae import init_models, compute_vae_loss_keep_dims
from src.anomalydetection.train import vae_loss_function
from src.classification.eval import init_dataloader
from src.config import DatasetConfig


def main(cfg: DatasetConfig, threshold: float):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    loader = init_dataloader(
        batch_size=4,
        nw=num_workers,
        path=cfg.get_classifier_dataset_split('validation'),
        transform=cfg.get_classifier_transform(),
        pin_memory=device == 'cuda',
    )

    model, vae_model = init_models(device, cfg.get_classifier_save_path(), cfg.get_vae_save_path())

    attack = getattr(torchattacks, cfg.attack_name.upper())(model)
    hm_transform = cfg.get_vae_transform()

    target_layers = [model.backbone.layer4[-1]]

    gt_labels = []
    predicted = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for img, label in tqdm(loader, desc='Processing'):
            img, label = img.to(device), label.to(device)

            grayscale_cam = torch.tensor(cam(input_tensor=img)).unsqueeze(1).to(device)
            grayscale_cam = hm_transform(grayscale_cam)

            adv_img = attack(img, label)
            grayscale_cam_adv = torch.tensor(cam(input_tensor=adv_img)).unsqueeze(1).to(device)
            grayscale_cam_adv = hm_transform(grayscale_cam_adv)

            recon, mu, logvar = vae_model(grayscale_cam)
            recon_adv, mu_adv, logvar_adv = vae_model(grayscale_cam_adv)

            score_clean = compute_vae_loss_keep_dims(recon, grayscale_cam, mu, logvar)
            score_adv = compute_vae_loss_keep_dims(recon_adv, grayscale_cam_adv, mu_adv, logvar_adv)
            print(score_clean)
            print(score_adv)

            all_scores = torch.cat([score_clean, score_adv])

            predicted.append((all_scores > threshold).int())
            gt_labels.append(torch.cat([torch.zeros_like(score_clean), torch.ones_like(score_adv)]))

    gt_labels = torch.cat(gt_labels)
    predicted = torch.cat(predicted)

    print((gt_labels == predicted).int().count_nonzero().item() / len(loader.dataset))

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Detection using VAE ELBO Scores")

    parser.add_argument("dataset", type=str, choices=["deepfake", "dogs-vs-cats"])
    parser.add_argument("attack", type=str, choices=["FGSM", "PGD"])
    parser.add_argument("--eps", type=float, default=0.03)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = DatasetConfig(args.dataset, args.attack)
    main(cfg, 41.1312)
