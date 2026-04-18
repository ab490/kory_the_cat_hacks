# Datasets for the two training phases.

import json
import os
import random

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

from decode import encode_text
from transforms import fit_line, resize_page, to_tensor


class DetectionDataset(Dataset):
    """Full page -> (image tensor, binary text mask)."""

    def __init__(self, manifest_path, root_dir, size):
        with open(manifest_path) as f:
            self.pages = json.load(f)["pages"]
        self.root = root_dir
        self.size = size

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i):
        item = self.pages[i]
        img = Image.open(os.path.join(self.root, item["image"])).convert("L")
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2 in item["boxes"]:
            draw.rectangle((x1, y1, x2, y2), fill=255)
        img = resize_page(img, self.size)
        mask = mask.resize(self.size, Image.NEAREST)
        return to_tensor(img), (to_tensor(mask) > 0).float()


class LineDataset(Dataset):
    """Line crop -> (image tensor, target index sequence, target length).
    Adds light padding/jitter so the recognizer tolerates the slightly looser
    crops that come from the detector at inference time."""

    def __init__(self, manifest_path, root_dir, augment=True):
        with open(manifest_path) as f:
            self.samples = json.load(f)["samples"]
        self.root = root_dir
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        item = self.samples[i]
        img = Image.open(os.path.join(self.root, item["image"])).convert("L")
        if self.augment:
            # random white padding on all sides — mimics detector box slack
            pad = (random.randint(0, 4), random.randint(0, 3),
                   random.randint(0, 4), random.randint(0, 3))
            img = ImageOps.expand(img, border=pad, fill=255)
        target = torch.tensor(encode_text(item["text"]), dtype=torch.long)
        return {
            "image": to_tensor(fit_line(img)),
            "target": target,
            "target_len": torch.tensor(len(target), dtype=torch.long),
            "text": item["text"],
        }


def line_collate(batch):
    # CTC loss wants flattened targets + per-sample lengths
    images = torch.stack([b["image"] for b in batch])
    targets = torch.cat([b["target"] for b in batch]) if batch else torch.empty(0, dtype=torch.long)
    target_lens = torch.stack([b["target_len"] for b in batch])
    return {
        "images": images, "targets": targets, "target_lens": target_lens,
        "texts": [b["text"] for b in batch],
    }
