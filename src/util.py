from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder


def init_dataloader(
        path: str,
        batch_size: int,
        nw: int,
        transform,
        pin_memory: bool,
        shuffle: bool,
        target_class_name: str | None = None
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