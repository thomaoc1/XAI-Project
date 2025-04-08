import torch
from torch.utils.data import Dataset


class UnsupervisedHeatmapDataset(Dataset):
    def __init__(self, heatmaps: torch.Tensor, transform=None):
        self.heatmaps = heatmaps
        self.transform = transform

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        img = self.heatmaps[idx].unsqueeze(0)
        if self.transform:
            return self.transform(img)
        else:
            return img


