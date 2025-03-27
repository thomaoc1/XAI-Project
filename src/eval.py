import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from src.classifier import DeepFakeClassifier


def evaluate(path: str, batch_size: int, nw: int, dev: str):
    model = DeepFakeClassifier().to(dev)
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    model.eval()

    dataset = ImageFolder("dataset/deepfake-dataset/validation", transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, shuffle=False, pin_memory=dev == 'cuda')

    for img, label in loader:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            preds = model(img).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(preds, label)
            total_loss += loss

            activations = F.sigmoid(preds)
            predicted = (activations >= 0.5).float()

            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)

    final_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0
    evaluate("model.pt", batch_size=64, nw=num_workers, dev=device)





