"""
OCR Microservice - Stage 1
POST /ocr  : accepts an image, returns extracted text
GET  /health : liveness check
"""

import os
import io
import time
import torch
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms

from model import OCRNet
from denoiser import Denoiser
from noise import gaussian_noise, salt_and_pepper_noise, sidd_noise
from segment import segment_characters

app = Flask(__name__)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "ocrnet.pth")
DENOISER_PATH = os.path.join(os.path.dirname(__file__), "weights", "denoiser.pth")

# Chars74K label map: 62 classes (0-9, A-Z, a-z)
LABEL_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load OCR model
model = OCRNet(num_classes=62).to(device)
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"Loaded OCR weights from {WEIGHTS_PATH}")
else:
    print("WARNING: no OCR weights found - run train.py first")
model.eval()

# Load denoiser
denoiser = Denoiser().to(device)
denoiser_loaded = False
if os.path.exists(DENOISER_PATH):
    denoiser.load_state_dict(torch.load(DENOISER_PATH, map_location=device))
    denoiser.eval()
    denoiser_loaded = True
    print(f"Loaded denoiser weights from {DENOISER_PATH}")
else:
    print("INFO: no denoiser weights found - skipping denoising step")

char_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

denoise_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def load_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


def denoise_image(img):
    if not denoiser_loaded:
        return img
    tensor = denoise_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised = denoiser(tensor)
    arr = denoised.squeeze().cpu().numpy()

    if arr.std() < 0.05:
        return img
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def predict_text(img):
    crops = segment_characters(img)
    print(f"[DEBUG] Segmentation found {len(crops)} crops, image size={img.size}, mode={img.mode}", flush=True)
    if not crops:
        tensor = char_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = model(tensor).argmax(dim=1).item()
        return LABEL_MAP[idx]
    # Batch all crops in one forward pass for speed
    batch = torch.stack([char_transform(c) for c in crops]).to(device)
    with torch.no_grad():
        indices = model(batch).argmax(dim=1).tolist()
    return "".join(LABEL_MAP[i] for i in indices)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' field"}), 400

    image_bytes = request.files["image"].read()
    noise_profile = request.form.get("noise_profile", "none")

    t0 = time.time()
    img = load_image(image_bytes)

    # Apply noise profile for benchmarking
    if noise_profile in ("gaussian", "salt_and_pepper", "sidd"):
        tensor = denoise_transform(img).unsqueeze(0)
        if noise_profile == "gaussian":
            tensor = gaussian_noise(tensor)
        elif noise_profile == "salt_and_pepper":
            tensor = salt_and_pepper_noise(tensor)
        elif noise_profile == "sidd":
            tensor = sidd_noise(tensor, {"sigma": 0.08, "beta": 0.03})
        arr = (tensor.squeeze().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Step 1: denoise full image
    img = denoise_image(img)

    # Step 2: segment + classify each character
    text = predict_text(img)
    latency_ms = (time.time() - t0) * 1000

    return jsonify({
        "text": text,
        "num_chars": len(text),
        "noise_profile": noise_profile,
        "denoised": denoiser_loaded,
        "latency_ms": round(latency_ms, 2),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)