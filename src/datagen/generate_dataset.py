import argparse

import torch
from pytorch_grad_cam import GradCAM
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.classification.binary_classifier import BinaryClassifier
from src.config import DatasetConfig


def init_dataloader(path: str, batch_size: int, nw: int, transform, pin_memory: bool, target_class_name: str | None = None):
    dataset = ImageFolder(path, transform=transform)

    if target_class_name:
        target_class_idx = dataset.class_to_idx[target_class_name]
        indices = [i for i, (_, label) in enumerate(dataset.samples) if label == target_class_idx]
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=pin_memory
    )


def main(cfg: DatasetConfig, batch_size: int, target_class_name: str | None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0

    loader = init_dataloader(
        cfg.get_classifier_dataset_split('train'),
        batch_size=batch_size,
        nw=num_workers,
        transform=cfg.get_classifier_transform(),
        pin_memory=device == 'cuda'
    )

    model = BinaryClassifier()
    model.load_state_dict(torch.load(cfg.get_classifier_save_path(), map_location=device, weights_only=True))
    model.eval()

    target_layers = [model.backbone.layer4[-1]]

    heatmaps = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for img, label in tqdm(loader, desc="Processing batches", unit="batch"):
            img, label = img.to(device), label.to(device)
            grayscale_cam = cam(input_tensor=img)
            batch_heatmaps = torch.tensor(grayscale_cam)
            heatmaps.append(batch_heatmaps)

    torch.save({
        'heatmaps': torch.cat(heatmaps, dim=0),
        },
        cfg.get_heatmap_dataset_path()
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Heatmap Dataset")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_class', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = DatasetConfig(args.dataset)
    main(cfg=config, batch_size=args.batch_size, target_class_name=args.target_class)
