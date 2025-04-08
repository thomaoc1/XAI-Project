import argparse

import cv2
import numpy as np
from skimage.feature import hog
import torch
from pytorch_grad_cam import GradCAM
from tqdm import tqdm

from src.classification.binary_classifier import BinaryClassifier
from src.classification.train import init_dataloader
from src.config import DatasetConfig


def extract_heatmap_features(correct_heatmaps: np.ndarray):
    all_features = []  # Will store the combined feature vectors

    for heatmap in correct_heatmaps:
        heatmap_as_img = (heatmap * 255.0).astype(np.uint8)

        # 1. Hu Moments (7 features)
        moments = cv2.moments(heatmap_as_img)
        hu_moments = cv2.HuMoments(moments).flatten()

        # 2. HOG Features
        resized = cv2.resize(heatmap, (64, 64))
        hog_feat = hog(
            resized,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            orientations=8
        )

        # 3. Intensity statistics (2 features)
        mean_intensity = np.array([np.mean(heatmap)])
        std_intensity = np.array([np.std(heatmap)])

        # Combine all features into one vector
        combined_features = np.concatenate(
            [
                hu_moments,
                hog_feat,
                mean_intensity,
                std_intensity
            ]
        )

        all_features.append(combined_features)

    return torch.tensor(np.stack(all_features, axis=0), dtype=torch.float32)


def main(cfg: DatasetConfig, batch_size: int):
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
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = DatasetConfig(args.dataset)
    main(cfg=config, batch_size=args.batch_size)
