from typing import Optional

import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder

from src.classification.binary_classifier import BinaryClassifier
from src.config import DatasetConfig


def init_dataloader(
        path: str,
        batch_size: int,
        nw: int,
        transform,
        pin_memory: bool,
        shuffle: bool,
        target_class_name: Optional[str] = None
):
    dataset = ImageFolder(path, transform=transform)

    if target_class_name:
        target_class_idx = dataset.class_to_idx[target_class_name]
        indices = [i for i, (_, label) in enumerate(dataset.samples) if label == target_class_idx]
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

def load_classifier(classifier_path: str, device: str):
    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.eval()
    return model