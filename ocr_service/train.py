# Train OCRNet: EMNIST Balanced (47 classes) + synthetic printed chars, noise on EMNIST.
# Usage: python train.py --epochs 25 --batch-size 128

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np

from model import OCRNet
from noise import gaussian_noise, salt_and_pepper_noise
from printed_dataset import PrintedCharDataset

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "ocrnet.pth")
METRICS_PATH = os.path.join(os.path.dirname(
    __file__), "weights", "metrics.json")

# label order from EMNIST "balanced" split (NIST)
LABEL_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
NUM_CLASSES = 47


def _rotate_image_tensor_clockwise_90_degrees(image_tensor):
    return torch.rot90(image_tensor, k=3, dims=[1, 2])


def _horizontal_flip_image_tensor(image_tensor):
    return image_tensor.flip(2)


class NoisyEMNIST(Dataset):
    def __init__(self, base_dataset, train=True):
        self.data = base_dataset
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, sample_index):
        image, label = self.data[sample_index]
        if self.train:
            choice = np.random.randint(0, 2)
            if choice == 0:
                image = gaussian_noise(image)
            else:
                image = salt_and_pepper_noise(image)
        return image, label


def evaluate(model, loader, device, noise_fn=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            if noise_fn is not None:
                images = torch.stack([noise_fn(img) for img in images])
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # EMNIST files are transposed; rot90 + flip matches upright text
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_rotate_image_tensor_clockwise_90_degrees),
        transforms.Lambda(_horizontal_flip_image_tensor),
    ])

    train_base = datasets.EMNIST(
        "../data", split="balanced", train=True, download=True, transform=transform
    )
    val_base = datasets.EMNIST(
        "../data", split="balanced", train=False, download=True, transform=transform
    )

    emnist_train = NoisyEMNIST(train_base, train=True)
    printed_train = PrintedCharDataset(length=len(train_base), train=True)
    train_dataset = ConcatDataset([emnist_train, printed_train])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_base, batch_size=256,
                            shuffle=False, num_workers=2)
    printed_val = PrintedCharDataset(length=5000, train=False)
    printed_loader = DataLoader(
        printed_val, batch_size=256, shuffle=False, num_workers=2)

    model = OCRNet(num_character_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.NLLLoss()

    epoch_log = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        emnist_acc = evaluate(model, val_loader, device)
        printed_acc = evaluate(model, printed_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  emnist={emnist_acc:.4f}  printed={printed_acc:.4f}")

        epoch_log.append({
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "emnist_acc": round(emnist_acc, 6),
            "printed_acc": round(printed_acc, 6),
        })
        with open(METRICS_PATH, "w") as f:
            json.dump({"epoch_log": epoch_log}, f, indent=2)

        if printed_acc > best_acc:
            best_acc = printed_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            epoch_log[-1]["best"] = True
            with open(METRICS_PATH, "w") as f:
                json.dump({
                    "epoch_log": epoch_log,
                    "best_epoch": epoch,
                    "best_printed_acc": round(best_acc, 6),
                }, f, indent=2)

    print("\nEvaluating per noise profile...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    metrics = {
        "emnist_clean": evaluate(model, val_loader, device),
        "emnist_gaussian": evaluate(model, val_loader, device, noise_fn=gaussian_noise),
        "emnist_salt_and_pepper": evaluate(model, val_loader, device, noise_fn=salt_and_pepper_noise),
        "printed_clean": evaluate(model, printed_loader, device),
    }

    for profile, acc in metrics.items():
        print(f"  {profile:25s}: {acc:.4f} ({acc*100:.2f}%)")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWeights saved to {WEIGHTS_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(args)
