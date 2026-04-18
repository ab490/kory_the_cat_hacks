# Train detector (BCE on text mask) or recognizer (CTC on line crops).
#
#   python train.py --stage detector   --manifest data/pages_manifest.json  --data_root data --epochs 20
#   python train.py --stage recognizer --manifest data/lines_manifest.json  --data_root data --epochs 30

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import PAGE_SIZE
from data import DetectionDataset, LineDataset, line_collate
from decode import decode_ctc
from model import CRNN, DBNet


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_detector(args, dev):
    ds = DetectionDataset(args.manifest, args.data_root, PAGE_SIZE)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    model = DBNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for img, mask in loader:
            img, mask = img.to(dev), mask.to(dev)
            logits = model(img)
            loss = F.binary_cross_entropy_with_logits(logits, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        print(f"[detector] epoch {epoch} loss {total / len(loader):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "detector.pt"))


def train_recognizer(args, dev):
    ds = LineDataset(args.manifest, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=line_collate)
    model = CRNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            images = batch["images"].to(dev)
            targets = batch["targets"].to(dev)
            target_lens = batch["target_lens"].to(dev)
            logits = model(images)                                  # [T, B, C]
            log_probs = F.log_softmax(logits, dim=-1)
            input_lens = torch.full((images.size(0),), logits.size(0),
                                    dtype=torch.long, device=dev)
            loss = ctc(log_probs, targets, input_lens, target_lens)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        # show one sample prediction each epoch to eyeball progress
        with torch.no_grad():
            sample = decode_ctc(model(images).detach())[0]
        print(f"[recognizer] epoch {epoch} loss {total / len(loader):.4f} "
              f"| sample: {batch['texts'][0]!r} -> {sample!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "recognizer.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["detector", "recognizer"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", default="weights")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    dev = device()
    print(f"device: {dev}")
    if args.stage == "detector":
        train_detector(args, dev)
    else:
        train_recognizer(args, dev)


if __name__ == "__main__":
    main()
