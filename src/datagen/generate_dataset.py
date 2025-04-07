import cv2
import numpy as np
from skimage.feature import hog
import torch
import torchattacks
from pytorch_grad_cam import GradCAM
from tqdm import tqdm

from src.classifier.binary_classifier import BinaryClassifier
from src.classifier.train import init_dataloader

VANILLA = 0
ADVERSARIAL = 1


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


def main(model_path: str, batch_size: int, adv: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0
    loader = init_dataloader(batch_size=batch_size, dev=device, nw=num_workers)

    model = BinaryClassifier()
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

            grayscale_cam = cam(input_tensor=img)
            vanilla_preds: torch.Tensor = cam.outputs.argmax(dim=1)

            if adv:
                adv_img = attack(img, label)
                grayscale_cam_adv = cam(input_tensor=adv_img)
                adv_preds: torch.Tensor = cam.outputs.argmax(dim=1)

                batch_ground_truths = torch.cat([label, label])
                batch_vanilla_preds = torch.cat([vanilla_preds, vanilla_preds])
                batch_heatmaps = torch.cat([torch.tensor(grayscale_cam), torch.tensor(grayscale_cam_adv)])
                batch_adv_preds = torch.cat([torch.ones(batch_size, dtype=torch.int) * -1, adv_preds])
                batch_labels = torch.cat(
                    [
                        torch.zeros(batch_size, dtype=torch.int),  # VANILLA
                        torch.ones(batch_size, dtype=torch.int)  # ADVERSARIAL
                    ]
                )

                all_vanilla_preds.append(batch_vanilla_preds)
                all_adv_preds.append(batch_adv_preds)
                all_ground_truths.append(batch_ground_truths)
                all_labels.append(batch_labels)
            else:
                batch_heatmaps = torch.tensor(grayscale_cam)

            heatmaps.append(batch_heatmaps)

            metrics = {
                'Vanilla Acc': (vanilla_preds == label.cpu()).float().mean().item()
            }

            if adv:
                metrics.update(
                    {
                        'Adv Acc': (adv_preds == label.cpu()).float().mean().item(),
                        'Flip Rate': (vanilla_preds != adv_preds).float().mean().item()
                    }
                )

            progress_bar.set_postfix(metrics)

    heatmaps = torch.cat(heatmaps, dim=0)
    obj = {
        'heatmaps': heatmaps,
    }

    if adv:
        heatmap_labels = torch.cat(all_labels, dim=0)
        classification = torch.cat(all_vanilla_preds, dim=0)
        ground_truths = torch.cat(all_ground_truths, dim=0)
        adv_classification = torch.cat(all_adv_preds, dim=0)
        obj.update(
            {
                'heatmaps': heatmaps,
                'labels': heatmap_labels,
                'model_vanilla_preds': classification,
                'ground_truth': ground_truths,
                'model_adv_preds': adv_classification,
            }
        )

    torch.save(
        obj, 'deep_fake_hm_dataset.pt'
    )

if __name__ == '__main__':
    main(model_path='new_model.pt', batch_size=64)










