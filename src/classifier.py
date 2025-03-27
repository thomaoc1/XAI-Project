import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class DeepFakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self._backbone.fc = nn.Linear(self._backbone.fc.in_features, 1)

    def forward(self, x):
        return self._backbone(x)
