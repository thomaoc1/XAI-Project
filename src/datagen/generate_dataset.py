import argparse
from typing import Optional

import torch
from pytorch_grad_cam import GradCAM
from sklearn.utils import shuffle
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import DatasetConfig
from src.util import init_dataloader, load_classifier


def main(cfg: DatasetConfig, batch_size: int, target_class_name: Optional[str]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    loader = init_dataloader(
        cfg.get_classifier_dataset_split('train'),
        batch_size=batch_size,
        nw=num_workers,
        transform=cfg.get_classifier_transform(),
        pin_memory=device == 'cuda',
        target_class_name=target_class_name,
        shuffle=False,
    )

    model = load_classifier(cfg.get_classifier_save_path(), device)
    target_layers = [model.backbone.layer4[-1]]

    heatmaps = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for img, label in tqdm(loader, desc="Processing batches", unit="batch"):
            img, label = img.to(device), label.to(device)
            grayscale_cam = cam(input_tensor=img)
            batch_heatmaps = torch.tensor(grayscale_cam)
            heatmaps.append(batch_heatmaps)

    torch.save(
        {
            'heatmaps': torch.cat(heatmaps, dim=0),
        },
        cfg.get_heatmap_dataset_path()
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Heatmap Dataset")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_class', type=str, default=None, choices=['fake', 'real', 'cat', 'dog'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = DatasetConfig(args.dataset, target_class=args.target_class)
    main(cfg=config, batch_size=args.batch_size, target_class_name=args.target_class)
