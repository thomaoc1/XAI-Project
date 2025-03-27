import matplotlib.pyplot as plt
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


if __name__ == '__main__':
    data = ImageFolder("dataset/deepfake-dataset/train", transform=torchvision.transforms.ToTensor())
    loader = DataLoader(data)

    for X, y in loader:
        plt.imshow(X.squeeze().permute(1, 2, 0))
        plt.show()
        break
    print("Success!")


