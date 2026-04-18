"""
Train the DnCNN denoiser on SimulatedNoisyOffice paired images.
"""

import argparse
import os
import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from denoiser import Denoiser

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "denoiser.pth")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "weights", "denoiser_metrics.json")


def parse_pair(noisy_filename):
    """
    Convert noisy filename to clean filename.
    Fontfre_Noisec_TR.png → Fontfre_Clean_TR.png
    """
    match = re.match(r"(Font\w+?)_Noise\w_(\w+\.png)", noisy_filename)
    if match:
        return f"{match.group(1)}_Clean_{match.group(2)}"
    return None


class DocumentPairDataset(Dataset):
    def __init__(self, data_dir, partition):
        """
        partition: 'TR', 'VA', or 'TE'
        """
        self.noisy_dir = os.path.join(data_dir, "simulated_noisy_images_grayscale")
        self.clean_dir = os.path.join(data_dir, "clean_images_grayscale")
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
        ])

        self.pairs = []
        for fname in os.listdir(self.noisy_dir):
            if not fname.endswith(f"_{partition}.png"):
                continue
            clean_fname = parse_pair(fname)
            if clean_fname is None:
                continue
            clean_path = os.path.join(self.clean_dir, clean_fname)
            if not os.path.exists(clean_path):
                continue
            self.pairs.append((
                os.path.join(self.noisy_dir, fname),
                clean_path,
            ))

        print(f"  [{partition}] Found {len(self.pairs)} noisy/clean pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
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
            # Resize clean to match output if shapes differ
            if output.shape != clean.shape:
                clean = torch.nn.functional.interpolate(
                    clean, size=output.shape[2:], mode="bilinear", align_corners=False
                )
            total_loss += criterion(output, clean).item()
    return total_loss / len(loader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DocumentPairDataset(args.data_dir, "TR")
    val_dataset = DocumentPairDataset(args.data_dir, "VA")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = Denoiser(num_layers=10, num_features=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    epoch_log = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            if output.shape != clean.shape:
                clean = torch.nn.functional.interpolate(
                    clean, size=output.shape[2:], mode="bilinear", align_corners=False
                )
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train_loss:.6f}  val_loss={val_loss:.6f}")

        entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 8),
            "val_loss": round(val_loss, 8),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            entry["best"] = True
            print(f"  → New best saved (val_loss={val_loss:.6f})")

        epoch_log.append(entry)
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "epoch_log": epoch_log,
                "best_val_loss": round(best_val_loss, 8),
                "best_epoch": next(e["epoch"] for e in epoch_log if e.get("best")),
            }, f, indent=2)

    print(f"\nDenoiser weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to SimulatedNoisyOffice folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(args)
