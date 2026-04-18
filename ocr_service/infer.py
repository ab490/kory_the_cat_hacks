import cv2
import numpy as np
import torch
from PIL import Image

from config import PAGE_SIZE
from decode import decode_ctc
from model import CRNN, DBNet
from transforms import fit_line, resize_page, to_tensor


def load_models(det_path, rec_path, dev):
    det, rec = DBNet().to(dev), CRNN().to(dev)
    sd_d = torch.load(det_path, map_location=dev)
    sd_r = torch.load(rec_path, map_location=dev)
    det.load_state_dict(sd_d["model"] if isinstance(sd_d, dict) and "model" in sd_d else sd_d)
    rec.load_state_dict(sd_r["model"] if isinstance(sd_r, dict) and "model" in sd_r else sd_r)
    det.eval(); rec.eval()
    return det, rec


def detect_boxes(detector, pil, dev, thresh=0.35):
    W0, H0 = pil.size
    tensor = to_tensor(resize_page(pil, PAGE_SIZE)).unsqueeze(0).to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(detector(tensor))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)
    mask = ((prob >= thresh) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, W0 // 80), 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 5 or w * h < 80:
            continue
        boxes.append((max(0, x - 3), max(0, y - 3),
                      min(W0, x + w + 3), min(H0, y + h + 3)))
    boxes.sort(key=lambda b: (b[1] // 10, b[0]))
    return boxes


def recognize(recognizer, pil_crop, dev):
    tensor = to_tensor(fit_line(pil_crop)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = recognizer(tensor)
    return decode_ctc(logits)[0]


def ocr_image(image_path, detector, recognizer, dev):
    pil = Image.open(image_path).convert("L")
    boxes = detect_boxes(detector, pil, dev) or [(0, 0, pil.width, pil.height)]
    lines = [recognize(recognizer, pil.crop(b), dev) for b in boxes]
    return "\n".join(lines), boxes
