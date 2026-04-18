# Very small synthetic page generator. Each page is a paragraph of random
# office-y words rendered in PIL, with noise/blur applied on top. We write:
#   pages/*.png         + pages_manifest.json (detector training)
#   line_crops/*.png    + lines_manifest.json (recognizer training)

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import PAGE_SIZE

WORDS = (
    "account analysis archive balance budget client contract department "
    "document estimate finance invoice ledger manager meeting office "
    "policy printed project quality receipt record report review schedule "
    "service shipping summary system total transfer vendor version "
    "the and for with from that this of in to on by as it is a an be"
).split()

# Try a few standard font paths; fall back to PIL's default bitmap font.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/Library/Fonts/Times New Roman.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def random_line(max_words=12):
    return " ".join(random.choice(WORDS) for _ in range(random.randint(4, max_words)))


def degrade(img):
    # Add mild blur + gaussian noise to imitate scanner artifacts.
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.normal(0, random.uniform(2, 10), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_page(size, n_lines=9):
    W, H = size
    img = Image.new("L", size, 245)
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    line_h = 20
    boxes, lines = [], []
    y = 12
    for _ in range(n_lines):
        if y + line_h > H - 6:
            break
        text = random_line()
        # clip the text to fit horizontally
        while font.getlength(text) > W - 16 and " " in text:
            text = text.rsplit(" ", 1)[0]
        draw.text((8, y), text, font=font, fill=0)
        bbox = (8, y, 8 + int(font.getlength(text)), y + line_h - 4)
        boxes.append(list(bbox))
        lines.append({"box": list(bbox), "text": text})
        y += line_h
    return degrade(img), boxes, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_pages", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    root = args.output_dir
    os.makedirs(f"{root}/pages", exist_ok=True)
    os.makedirs(f"{root}/line_crops", exist_ok=True)

    pages_manifest, lines_manifest = [], []
    for i in range(args.num_pages):
        img, boxes, lines = render_page(PAGE_SIZE)
        p = f"pages/page_{i:05d}.png"
        img.save(f"{root}/{p}")
        pages_manifest.append({"image": p, "boxes": boxes})
        for j, ln in enumerate(lines):
            x1, y1, x2, y2 = ln["box"]
            crop = img.crop((max(0, x1 - 2), max(0, y1 - 2), x2 + 2, y2 + 2))
            q = f"line_crops/line_{i:05d}_{j:02d}.png"
            crop.save(f"{root}/{q}")
            lines_manifest.append({"image": q, "text": ln["text"]})

    with open(f"{root}/pages_manifest.json", "w") as f:
        json.dump({"pages": pages_manifest}, f)
    with open(f"{root}/lines_manifest.json", "w") as f:
        json.dump({"samples": lines_manifest}, f)
    print(f"wrote {args.num_pages} pages, {len(lines_manifest)} line crops to {root}")


if __name__ == "__main__":
    main()
