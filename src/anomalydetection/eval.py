import numpy as np
import torch
import argparse
import torchattacks
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from pytorch_grad_cam import GradCAM

from src.anomalydetection.vae import CNNVAE
from src.classification.binary_classifier import BinaryClassifier
from src.classification.eval import init_dataloader


def compute_elbo(x, recon, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction='none').flatten(1).mean(1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon_loss + kl

def init_models(device: str, classifier_path: str, vae_model_path: str):
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.eval()

    vae_model = CNNVAE(latent_dim=128).to(device)
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=device, weights_only=True))
    vae_model.eval()

    return classifier, vae_model

def evaluate(score_clean: torch.Tensor, score_adv: torch.Tensor):
    scores_all = torch.cat([score_clean, score_adv]).numpy()
    labels_all = np.concatenate([np.zeros(len(score_clean)), np.ones(len(score_adv))])

    fpr, tpr, thresholds = roc_curve(labels_all, scores_all)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    best_threshold = thresholds[np.argmax(j_scores)]

    print(f'Clean Mean:      {score_clean.mean().item():.4f}, Std: {score_clean.std().item():.4f}')
    print(f'Adv   Mean:      {score_adv.mean().item():.4f}, Std: {score_adv.std().item():.4f}')
    print(f'AUC   Score:     {roc_auc}')
    print(f'Best  Threshold: {best_threshold}')

    return roc_auc, best_threshold, fpr, tpr


def save_auc_plot(roc_auc, fpr, tpr, path='figs/vae_auc_plot.png'):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random guess)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — VAE ELBO Detector')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(path)


def save_results(all_score_clean: torch.Tensor, all_score_adv: torch.Tensor, roc_auc: float, best_threshold: float, path: str):
    torch.save(
        {
            'all_score_clean': all_score_clean,
            'all_score_adv': all_score_adv,
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
        }, path
    )


def main(run_name: str, dataset_path: str, classifier_path: str, vae_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    loader = init_dataloader(batch_size=64, dev=device, nw=num_workers, path=dataset_path)
    model, vae_model = init_models(device, classifier_path, vae_path)
    attack = torchattacks.FGSM(model)

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

            score_clean = compute_elbo(grayscale_cam, recon, mu, logvar)
            score_adv = compute_elbo(grayscale_cam_adv, recon_adv, mu_adv, logvar_adv)

            all_score_clean.append(score_clean.detach().cpu())
            all_score_adv.append(score_adv.detach().cpu())

    all_score_clean = torch.cat(all_score_clean)
    all_score_adv = torch.cat(all_score_adv)

    roc_auc, best_threshold, fpr, tpr = evaluate(all_score_clean, all_score_adv)
    save_results(all_score_clean, all_score_adv, roc_auc, best_threshold, path=f'results/{run_name}.pt')
    save_auc_plot(roc_auc, fpr, tpr, path=f'results/figs/{run_name}.png')


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Detection using VAE ELBO Scores")

    # Required arguments
    parser.add_argument("--run_name", "-rn", type=str, required=True, help="Name for this run")
    parser.add_argument("--dataset_path", "-dp", type=str, required=True, help="Path to the dataset directory")

    # Model paths (with defaults)
    parser.add_argument("--classifier_path", "-cp", type=str, required=True, help="Path to classifier model")
    parser.add_argument("--vae_model_path", "-vp", type=str, required=True, help="Path to VAE model")

    # Attack configuration
    parser.add_argument("--attack", type=str, default="FGSM", choices=["FGSM", "PGD", "CW"])
    parser.add_argument("--eps", type=float, default=0.03, help="Attack perturbation magnitude (default: 0.03)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        run_name=args.run_name,
        dataset_path=args.dataset_path,
        classifier_path=args.classifier_path,
        vae_path=args.vae_model_path,
    )