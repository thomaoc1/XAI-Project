import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import classification_report

from src.classification.binary_classifier import BinaryClassifier

def init_dataloader(batch_size, nw, dev, path="dataset/deepfake-dataset/validation"):
    dataset = ImageFolder(path, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, shuffle=False, pin_memory=dev == 'cuda')
    return loader


def evaluate(path: str, batch_size: int, nw: int, dev: str):
    model = BinaryClassifier().to(dev)
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    model.eval()

    loader = init_dataloader(batch_size=batch_size, nw=nw, dev=dev)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    progress_bar = tqdm.tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for img, label in progress_bar:
            img, label = img.to(dev), label.to(dev)
            preds: torch.Tensor = model(img).squeeze(1)
            loss = F.cross_entropy(preds, label)
            total_loss += loss.item()

            predicted = preds.argmax(dim=1)

            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if device == 'cuda' else 0
    evaluate("new_model.pt", batch_size=64, nw=num_workers, dev=device)
