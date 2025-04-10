import torch
from torch.utils.data import Dataset


class UnsupervisedHeatmapDataset(Dataset):
    def __init__(self, heatmaps: torch.Tensor):
        self.heatmaps = heatmaps

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        return self.heatmaps[idx].unsqueeze(0)


