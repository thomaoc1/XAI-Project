import argparse

import numpy as np
import torch
import torchattacks
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from src.anomalydetection.vae import CNNVAE
from src.classification.binary_classifier import BinaryClassifier
from src.config import DatasetConfig
from src.util import init_dataloader, load_classifier


def compute_vae_loss_keep_dims(recon, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction='none').flatten(1).mean(1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon_loss + kl


def init_models(device: str, classifier_path: str, vae_model_path: str):
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.eval()
    classifier = load_classifier(classifier_path, device)

    vae_model = CNNVAE().to(device)
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=device, weights_only=True))
    vae_model.eval()

    return classifier, vae_model


def evaluate(score_clean: torch.Tensor, score_adv: torch.Tensor):
    scores_all = torch.cat([score_clean, score_adv]).numpy()
    labels_all = np.concatenate([np.zeros(len(score_clean)), np.ones(len(score_adv))])

    fpr, tpr, thresholds = roc_curve(labels_all, scores_all)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    best_threshold = thresholds[optimal_idx]

    predicted = (scores_all > best_threshold).astype(int)
    accuracy = np.sum(predicted == labels_all) / len(labels_all)

    print(f'Clean Mean:      {score_clean.mean().item():.2f}, Std: {score_clean.std().item():.2f}')
    print(f'Adv   Mean:      {score_adv.mean().item():.2f}, Std: {score_adv.std().item():.2f}')
    print(f'AUC   Score:     {roc_auc:.2f}')
    print(f'Best  Threshold: {best_threshold:.2f}')
    print(f"Accuracy with threshold: {accuracy:.4f}")

    return roc_auc, thresholds, fpr, tpr, optimal_idx, accuracy


def save_auc_plot(roc_auc, fpr, tpr, thresholds, optimal_idx, path='figs/vae_auc_plot.png'):
    # Find best threshold (maximize TPR - FPR)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Mark the optimal threshold
    plt.scatter(optimal_fpr, optimal_tpr, color='red', label=f'Best Threshold = {optimal_threshold:.4f}', zorder=5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    print(f'Saved plot to {path}')
    plt.savefig(path)
    plt.close()


def save_results(
        all_score_clean: torch.Tensor,
        all_score_adv: torch.Tensor,
        roc_auc: float,
        best_threshold: float,
        accuracy: float,
        path: str
    ):
    torch.save(
        {
            'all_score_clean': all_score_clean,
            'all_score_adv': all_score_adv,
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
        }, path
    )


def main(cfg: DatasetConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    loader = init_dataloader(
        batch_size=64,
        nw=num_workers,
        path=cfg.get_classifier_dataset_split('validation'),
        transform=cfg.get_classifier_transform(),
        pin_memory=device == 'cuda',
        shuffle=False,
        target_class_name=cfg.target_class,
    )

    model, vae_model = init_models(device, cfg.get_classifier_save_path(), cfg.get_vae_save_path())

    attack = getattr(torchattacks, cfg.attack_name.upper())(model)

    target_layers = [model.backbone.layer4[-1]]

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

            score_clean = compute_vae_loss_keep_dims(recon, grayscale_cam, mu, logvar)
            score_adv = compute_vae_loss_keep_dims(recon, grayscale_cam_adv, mu_adv, logvar_adv)

            all_score_clean.append(score_clean.detach().cpu())
            all_score_adv.append(score_adv.detach().cpu())

    all_score_clean = torch.cat(all_score_clean)
    all_score_adv = torch.cat(all_score_adv)

    roc_auc, thresholds, fpr, tpr, optimal_idx, accuracy = evaluate(all_score_clean, all_score_adv)
    save_results(
        all_score_clean,
        all_score_adv,
        roc_auc,
        thresholds[optimal_idx],
        accuracy,
        path=cfg.get_vae_results_save_path(),
    )

    save_auc_plot(
        roc_auc,
        fpr,
        tpr,
        thresholds,
        optimal_idx,
        path=cfg.get_vae_figs_save_path(),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Detection using VAE ELBO Scores")

    parser.add_argument("dataset", type=str, choices=["deepfake", "dogs-vs-cats"])
    parser.add_argument('--target_class', type=str, default=None, choices=['fake', 'real', 'cat', 'dog'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = DatasetConfig(args.dataset, attack_name='FGSM', target_class=args.target_class)

    main(
        cfg=config,
    )
