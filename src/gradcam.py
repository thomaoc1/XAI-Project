import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchattacks

from src.classifier import DeepFakeClassifier
from src.train import init_dataloader

VANILLA = 0
ADVERSARIAL = 1

def extract_heatmap_features(correct_heatmaps: np.ndarray):
    all_hu_moments = []
    for heatmap in correct_heatmaps:
        heatmap_as_img = (heatmap * 255.0).astype(np.uint8)
        moments = cv2.moments(heatmap_as_img)
        hu_moments = cv2.HuMoments(moments).flatten()
        all_hu_moments.append(hu_moments)
    return torch.tensor(np.stack(all_hu_moments, axis=0))

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
    model_classifications = []
    ground_truths = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            ground_truths.append(label.cpu())

            # Original Image
            grayscale_cam = cam(input_tensor=img)
            extracted_heatmap_features = extract_heatmap_features(grayscale_cam)
            model_classifications.append(cam.outputs.argmax(dim=1))

            # Adv. Example
            adv_img = attack(img, label)
            grayscale_cam_adv = cam(input_tensor=adv_img)
            adv_extracted_heatmap_features = extract_heatmap_features(grayscale_cam_adv)

            heatmaps.append(extracted_heatmap_features)
            heatmaps.append(adv_extracted_heatmap_features)

    heatmaps = torch.concat(heatmaps, dim=0)
    heatmap_labels = (torch.arange(2 * len(loader)) % 2).repeat_interleave(batch_size)
    model_classifications = torch.concat(model_classifications, dim=0).repeat_interleave(2)
    ground_truths = torch.concat(ground_truths, dim=0).repeat_interleave(2)

if __name__ == '__main__':
    main(model_path='new_model.pt', batch_size=8)










