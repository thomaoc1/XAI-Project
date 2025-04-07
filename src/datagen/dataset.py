import torch
from torch.utils.data import Dataset


class UnsupervisedHeatmapDataset(Dataset):
    def __init__(self, heatmaps: torch.Tensor):
        self.heatmaps = heatmaps

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        return {
            'heatmaps': self.heatmaps[idx],
        }


class HeatmapDataset(UnsupervisedHeatmapDataset):
    def __init__(self, heatmaps: torch.Tensor, labels: torch.Tensor, vanilla_preds: torch.Tensor, ground_truth: torch.Tensor, adv_preds: torch.Tensor):
        super().__init__(heatmaps)

        self.adv_labels = labels
        self.vanilla_preds = vanilla_preds
        self.adv_preds = adv_preds
        self.ground_truth = ground_truth

    def __getitem__(self, idx):
        return {
            'heatmaps': self.heatmaps[idx],
            'labels': self.adv_labels[idx],
            'adv_preds': self.adv_preds[idx],
            'vanilla_preds': self.vanilla_preds[idx],
            'ground_truth': self.ground_truth[idx]
        }
