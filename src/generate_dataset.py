import cv2
import numpy as np
from skimage.feature import hog
import torch
from torch.utils.data.dataset import Dataset
import torchattacks
from pytorch_grad_cam import GradCAM
from tqdm import tqdm

from src.classifier import DeepFakeClassifier
from src.classifier.train import init_dataloader

VANILLA = 0
ADVERSARIAL = 1

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

def filter_heatmaps(cam, grayscale_cam, label):
    activations: torch.Tensor = cam.outputs
    predicted: torch.Tensor = activations.argmax(dim=1)
    correct_mask = (predicted != label).cpu().numpy()
    correct_heatmaps = grayscale_cam[~correct_mask]
    return correct_heatmaps

def main(model_path: str, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0
    loader = init_dataloader(batch_size=batch_size, dev=device, nw=num_workers)

    model = DeepFakeClassifier()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    attack = torchattacks.FGSM(model)

    target_layers = [model.backbone.layer4[-1]]

    heatmaps = []
    all_labels = []
    all_vanilla_preds = []
    all_adv_preds = []
    all_ground_truths = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        progress_bar = tqdm(loader, desc="Processing batches", unit="batch")

        for img, label in progress_bar:
            img, label = img.to(device), label.to(device)

            # Original Image
            grayscale_cam = cam(input_tensor=img)
            vanilla_preds: torch.Tensor = cam.outputs.argmax(dim=1)

            # Adv. Example
            adv_img = attack(img, label)
            grayscale_cam_adv = cam(input_tensor=adv_img)
            adv_preds: torch.Tensor = cam.outputs.argmax(dim=1)

            batch_ground_truths = torch.cat([label, label])
            batch_vanilla_preds = torch.cat([vanilla_preds, vanilla_preds])
            batch_adv_preds = torch.cat([torch.ones(batch_size, dtype=torch.int) * -1, adv_preds])
            batch_heatmaps = torch.cat([torch.tensor(grayscale_cam), torch.tensor(grayscale_cam_adv)])
            batch_labels = torch.cat(
                [
                    torch.zeros(batch_size, dtype=torch.int),  # VANILLA
                    torch.ones(batch_size, dtype=torch.int)  # ADVERSARIAL
                ]
            )

            heatmaps.append(batch_heatmaps)
            all_labels.append(batch_labels)
            all_vanilla_preds.append(batch_vanilla_preds)
            all_adv_preds.append(batch_adv_preds)
            all_ground_truths.append(batch_ground_truths)

            progress_bar.set_postfix(
                {
                    'Vanilla Acc': (vanilla_preds == label).float().mean().item(),
                    'Adv Acc': (adv_preds == label).float().mean().item(),
                    'Flip Rate': (vanilla_preds != adv_preds).float().mean().item()
                }
            )

    heatmaps = torch.cat(heatmaps, dim=0)
    heatmap_labels = torch.cat(all_labels, dim=0)
    classification = torch.cat(all_vanilla_preds, dim=0)
    ground_truths = torch.cat(all_ground_truths, dim=0)
    adv_classification = torch.cat(all_adv_preds, dim=0)

    print(heatmaps.shape)
    print(heatmap_labels)
    print(classification)
    print(ground_truths)
    print(adv_classification)

    torch.save(
        {
            'heatmaps': heatmaps,
            'labels': heatmap_labels,
            'model_vanilla_preds': classification,
            'ground_truth': ground_truths,
            'model_adv_preds': adv_classification,
        }, 'deep_fake_hm_dataset.pt'
    )

if __name__ == '__main__':
    main(model_path='new_model.pt', batch_size=64)










