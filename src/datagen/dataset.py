import torch
from torch.utils.data import Dataset


class HeatmapDataset(Dataset):
    def __init__(self, heatmaps: torch.Tensor, adv_labels: torch.Tensor,
                 model_preds: torch.Tensor, ground_truth: torch.Tensor):
        self.heatmaps = heatmaps
        self.adv_labels = adv_labels
        self.model_preds = model_preds
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        return {
            'features': self.heatmaps[idx],
            'adv_label': self.adv_labels[idx],
            'model_pred': self.model_preds[idx],
            'ground_truth': self.ground_truth[idx]
        }