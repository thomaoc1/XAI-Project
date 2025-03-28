import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from torchvision.transforms import ToTensor

from src.classifier import DeepFakeClassifier


class GradCAM:
    def __init__(self, model: nn.Module, target_layer):
        self.model = model
        model.eval()
        self.target_layer = target_layer
        self.hooks = []
        self.activations = None
        self.grads = None

    def _forward_hook(self, module, inp, out):
        self.activations = out.squeeze()

    def _backward_hook(self, module, grad_inp, grad_out):
        self.grads = grad_out[0].squeeze()

    def __enter__(self):
        self.hooks = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook)
        ]
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()

    def compute_heatmap(self, x: torch.Tensor, target_class: int):
        if not len(self.hooks):
            raise ValueError('Ensure that you are calling within GradCAM context to ensure hooks are initialised/freed')

        output = model(x)
        output[:, target_class].backward()

        importance_weights = torch.mean(self.grads, dim=[1, 2])
        print(self.activations.shape, importance_weights.shape)
        scaled_activations = self.activations * importance_weights.unsqueeze(1).unsqueeze(2)
        L = F.relu(scaled_activations.sum(dim=0))
        print(L.shape)

        original_height, original_width = x.shape[2], x.shape[3]
        upsampled_L = F.interpolate(
            L.unsqueeze(0).unsqueeze(0),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        )
        return upsampled_L.squeeze(0).squeeze(0), L

    @staticmethod
    def visualise_heatmap(x: torch.Tensor, upsampled_L: torch.Tensor):
        upsampled_L = upsampled_L.cpu().detach().numpy()

        upsampled_L = np.maximum(upsampled_L, 0)
        upsampled_L = upsampled_L - np.min(upsampled_L)
        if np.max(upsampled_L) != 0:
            upsampled_L = upsampled_L / np.max(upsampled_L)

        heatmap_uint8 = np.uint8(255 * upsampled_L)

        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        orig_img = x.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        orig_img = np.uint8(255 * orig_img)

        alpha = 0.4
        superimposed_img = (colored_heatmap * alpha + orig_img * (1 - alpha)).astype(np.uint8)

        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.title('Grad-CAM Heatmap')
        plt.show()


if __name__ == '__main__':
    img = Image.open("dataset/deepfake-dataset/validation/real/aybgughjxh_56_0.png").convert("RGB")
    tensor_img = ToTensor()(img).unsqueeze(0)

    model = DeepFakeClassifier()
    model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
    model.eval()

    with GradCAM(model, model.backbone.layer4) as G:
        heatmap, _ = G.compute_heatmap(tensor_img, 0)

    GradCAM.visualise_heatmap(tensor_img, heatmap)










