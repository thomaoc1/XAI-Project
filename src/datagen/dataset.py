import torch
from torch.utils.data import Dataset


class UnsupervisedHeatmapDataset(Dataset):
    def __init__(self, heatmaps: torch.Tensor):
        self.heatmaps = heatmaps
        self._is_normalised = heatmaps.max().item() <= 1.0

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        if self._is_normalised:
            return self.heatmaps[idx].unsqueeze(0)
        else:
            return self.heatmaps[idx].unsqueeze(0) / 255.0
