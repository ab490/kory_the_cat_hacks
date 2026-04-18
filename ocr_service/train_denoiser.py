# Train DnCNN: python train_denoiser.py --data-dir ../SimulatedNoisyOffice --epochs 50

import argparse
import os
import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from denoiser import Denoiser

WEIGHTS_PATH = os.path.join(os.path.dirname(
    __file__), "weights", "denoiser.pth")
METRICS_PATH = os.path.join(os.path.dirname(
    __file__), "weights", "denoiser_metrics.json")


def parse_pair(noisy_filename):
    match = re.match(r"(Font\w+?)_Noise\w_(\w+\.png)", noisy_filename)
    assert match is not None
    return match.group(1) + "_Clean_" + match.group(2)


class DocumentPairDataset(Dataset):
    def __init__(self, data_dir, partition):
        self.noisy_dir = os.path.join(
            data_dir, "simulated_noisy_images_grayscale")
        self.clean_dir = os.path.join(data_dir, "clean_images_grayscale")
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
        ])

        self.pairs = []
        for fname in os.listdir(self.noisy_dir):
            if not fname.endswith("_" + partition + ".png"):
                continue
            clean_fname = parse_pair(fname)
            clean_path = os.path.join(self.clean_dir, clean_fname)
            assert os.path.isfile(clean_path)
            self.pairs.append((
                os.path.join(self.noisy_dir, fname),
                clean_path,
            ))

        print("  [" + partition + "] " + str(len(self.pairs)) + " pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, sample_index):
        noisy_path, clean_path = self.pairs[sample_index]
        noisy = self.transform(Image.open(noisy_path))
        clean = self.transform(Image.open(clean_path))
        return noisy, clean


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            total_loss += criterion(output, clean).item()
    return total_loss / len(loader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    train_dataset = DocumentPairDataset(args.data_dir, "TR")
    val_dataset = DocumentPairDataset(args.data_dir, "VA")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    model = Denoiser(num_layers=10, num_features=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    epoch_log = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, criterion)
        print("Epoch " + str(epoch) + "/" + str(args.epochs) + "  train_loss=" +
              str(round(avg_train_loss, 6)) + "  val_loss=" + str(round(val_loss, 6)))

        entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 8),
            "val_loss": round(val_loss, 8),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), WEIGHTS_PATH)
            entry["best"] = True

        epoch_log.append(entry)
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "epoch_log": epoch_log,
                "best_val_loss": round(best_val_loss, 8),
                "best_epoch": best_epoch,
            }, f, indent=2)

    print("\nDenoiser weights -> " + WEIGHTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(args)
