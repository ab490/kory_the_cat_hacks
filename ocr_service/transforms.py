import numpy as np
import torch
from PIL import Image

from config import LINE_H, LINE_W


def to_tensor(pil):
    arr = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0)
    return (t - 0.5) / 0.5


def resize_page(pil, size):
    return pil.convert("L").resize(size, Image.BILINEAR)


def fit_line(pil, h=LINE_H, w=LINE_W):
    pil = pil.convert("L")
    orig_w, orig_h = pil.size
    if orig_h <= 0 or orig_w <= 0:
        return Image.new("L", (w, h), 255)
    new_w = max(1, min(w, int(round(orig_w * h / orig_h))))
    resized = pil.resize((new_w, h), Image.BILINEAR)
    canvas = Image.new("L", (w, h), 255)
    canvas.paste(resized, (0, 0))
    return canvas
