import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import tqdm
from sklearn.metrics import classification_report

from src.classification.binary_classifier import BinaryClassifier
from src.config import DatasetConfig
from src.util import init_dataloader


def main(cfg: DatasetConfig, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(cfg.get_classifier_save_path(), map_location=device, weights_only=True))
    model.eval()

    loader = init_dataloader(
        cfg.get_classifier_dataset_split('validation'),
        batch_size=batch_size,
        nw=num_workers,
        transform=cfg.get_classifier_transform(),
        pin_memory=device == 'cuda',
        shuffle=False,
        target_class_name=cfg.target_class,
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    progress_bar = tqdm.tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for img, label in progress_bar:
            img, label = img.to(device), label.to(device)
            preds: torch.Tensor = model(img).squeeze(1)
            loss = F.cross_entropy(preds, label)
            total_loss += loss.item()

            predicted = preds.argmax(dim=1)

            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=cfg.get_classes(), digits=4))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Classifier")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = DatasetConfig(args.dataset)
    main(
        cfg=config,
        batch_size=args.batch_size,
    )
