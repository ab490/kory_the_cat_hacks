"""
Train OCRNet on Chars74K Fnt dataset with document noise augmentation.

Noise augmentation applied online during training:
  - Gaussian noise
  - Salt-and-pepper noise
  - Real document noise patches from SimulatedNoisyOffice + RealNoisyOffice
"""

import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import OCRNet
from noise import gaussian_noise, salt_and_pepper_noise

DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data/English/Fnt")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "ocrnet.pth")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "weights", "metrics.json")

NOISY_DOC_DIRS = [
    os.path.join(os.path.dirname(__file__),
                 "../data/SimulatedNoisyOffice/simulated_noisy_images_grayscale"),
    os.path.join(os.path.dirname(__file__),
                 "../data/RealNoisyOffice/real_noisy_images_grayscale"),
]

LABEL_MAP  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = 62
PATCH_SIZE  = 28


def load_doc_patches(dirs, patch_size = PATCH_SIZE):
    patches = []
    for doc_dir in dirs:
        if not os.path.isdir(doc_dir):
            continue
        for fname in os.listdir(doc_dir):
            if not fname.endswith(".png"):
                continue
            try:
                img = Image.open(os.path.join(doc_dir, fname)).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                h, w = arr.shape
                for _ in range(30):
                    if h < patch_size or w < patch_size:
                        continue
                    r = random.randint(0, h - patch_size)
                    c = random.randint(0, w - patch_size)
                    patch = arr[r:r + patch_size, c:c + patch_size]
                    if patch.mean() > 0.5:
                        patches.append(patch)
            except Exception:
                continue
    print(f"  Loaded {len(patches)} document noise patches")
    return patches


def apply_doc_patch(tensor, patches):
    patch = patches[random.randint(0, len(patches) - 1)]
    patch_t = torch.from_numpy(patch).unsqueeze(0)          # (1, 28, 28)
    ink_mask = tensor < 0.5
    blended = patch_t.clone()
    blended[ink_mask] = tensor[ink_mask] * 0.4 + patch_t[ink_mask] * 0.1
    return torch.clamp(blended, 0, 1)


class Chars74KDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment = False,
                 doc_patches = None):
        self.samples = []
        self.transform = transform
        self.augment = augment
        self.doc_patches = doc_patches or []

        for class_idx in range(NUM_CLASSES):
            folder = os.path.join(data_dir, f"Sample{class_idx + 1:03d}")
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(folder, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)

        if self.augment:
            r = random.random()
            if r < 0.40 and self.doc_patches:
                # Real document background
                img = apply_doc_patch(img, self.doc_patches)
            elif r < 0.65:
                # Gaussian with random intensity
                std = random.uniform(0.05, 0.25)
                img = torch.clamp(img + torch.randn_like(img) * std, 0, 1)
            elif r < 0.90:
                # Salt-and-pepper with random density
                prob = random.uniform(0.02, 0.10)
                mask = torch.rand_like(img)
                img = img.clone()
                img[mask < prob / 2] = 0.0
                img[mask > 1 - prob / 2] = 1.0
            # else: clean (10%)

        return img, label


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
    return correct / total if total > 0 else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    doc_patches = load_doc_patches(NOISY_DOC_DIRS)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    full_dataset = Chars74KDataset(DATA_DIR, transform=transform,
                                   augment=False, doc_patches=doc_patches)
    print(f"Total samples: {len(full_dataset)}")

    val_size  = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_idx, val_idx = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Augmented train dataset, clean val dataset
    train_dataset = Chars74KDataset(DATA_DIR, transform=transform,
                                    augment=True, doc_patches=doc_patches)
    val_dataset   = Chars74KDataset(DATA_DIR, transform=val_transform,
                                    augment=False)

    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, list(train_idx))
    val_dataset   = Subset(val_dataset,   list(val_idx))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=256,
                              shuffle=False, num_workers=2)

    model     = OCRNet(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.NLLLoss()

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    epoch_log = []
    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        avg_loss    = total_loss / len(train_loader)
        clean_acc   = evaluate(model, val_loader, device)
        gauss_acc   = evaluate(model, val_loader, device, noise_fn=gaussian_noise)
        sp_acc      = evaluate(model, val_loader, device, noise_fn=salt_and_pepper_noise)

        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"clean={clean_acc:.4f}  gaussian={gauss_acc:.4f}  s&p={sp_acc:.4f}")

        entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "clean_acc": round(clean_acc, 6),
            "gaussian_acc": round(gauss_acc, 6),
            "salt_pepper_acc": round(sp_acc, 6),
        }
        if clean_acc > best_acc:
            best_acc = clean_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            entry["best"] = True
            print(f"  → New best: {best_acc:.4f}")

        epoch_log.append(entry)
        with open(METRICS_PATH, "w") as f:
            json.dump({"epoch_log": epoch_log,
                       "best_clean_acc": round(best_acc, 6)}, f, indent=2)

    print("\nEvaluating best model per noise profile...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    metrics = {
        "clean":           round(evaluate(model, val_loader, device), 6),
        "gaussian":        round(evaluate(model, val_loader, device,
                                          noise_fn=gaussian_noise), 6),
        "salt_and_pepper": round(evaluate(model, val_loader, device,
                                          noise_fn=salt_and_pepper_noise), 6),
    }
    for profile, acc in metrics.items():
        print(f"  {profile:20s}: {acc:.4f} ({acc*100:.2f}%)")

    with open(METRICS_PATH, "w") as f:
        json.dump({"epoch_log": epoch_log,
                   "best_clean_acc": round(best_acc, 6),
                   "final_metrics": metrics}, f, indent=2)

    print(f"\nWeights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(args)