# Stage 1: POST /ocr (image -> text), GET /health

import os
import io
import sys
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
DENOISER_PATH = os.path.join(os.path.dirname(
    __file__), "weights", "denoiser.pth")

LABEL_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
NUM_CLASSES = 47

SIDD_PARAMS = {"sigma": 0.08, "beta": 0.03}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isfile(WEIGHTS_PATH):
    sys.exit("missing ocr_service/weights/ocrnet.pth — run train.py")
if not os.path.isfile(DENOISER_PATH):
    sys.exit("missing ocr_service/weights/denoiser.pth — run train_denoiser.py")

model = OCRNet(num_character_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

denoiser = Denoiser().to(device)
denoiser.load_state_dict(torch.load(DENOISER_PATH, map_location=device))
denoiser.eval()

char_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

denoise_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def denoise_image(pil_image):
    grayscale_tensor = denoise_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_array = denoiser(grayscale_tensor).squeeze().cpu().numpy()
    return Image.fromarray((denoised_array * 255).astype(np.uint8))


def classify_crops(character_image_crops):
    if len(character_image_crops) == 0:
        return ""
    batch_tensor = torch.stack([char_transform(crop) for crop in character_image_crops]).to(device)
    with torch.no_grad():
        predicted_class_indices = model(batch_tensor).argmax(dim=1).tolist()
    recognized_text = ""
    for class_index in predicted_class_indices:
        recognized_text += LABEL_MAP[class_index]
    return recognized_text


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' field"}), 400

    image_bytes = request.files["image"].read()
    noise_profile = request.form.get("noise_profile", "none")

    request_start_time_seconds = time.time()
    working_image = Image.open(io.BytesIO(image_bytes))

    if noise_profile == "gaussian":
        noisy_tensor = denoise_transform(working_image).unsqueeze(0).to(device)
        noisy_tensor = gaussian_noise(noisy_tensor)
        working_image = Image.fromarray(
            (noisy_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))
    elif noise_profile == "salt_and_pepper":
        noisy_tensor = denoise_transform(working_image).unsqueeze(0).to(device)
        noisy_tensor = salt_and_pepper_noise(noisy_tensor)
        working_image = Image.fromarray(
            (noisy_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))
    elif noise_profile == "sidd":
        noisy_tensor = denoise_transform(working_image).unsqueeze(0).to(device)
        noisy_tensor = sidd_noise(noisy_tensor, SIDD_PARAMS)
        working_image = Image.fromarray(
            (noisy_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))

    working_image = denoise_image(working_image)
    recognized_text = classify_crops(segment_characters(working_image))

    latency_milliseconds = (time.time() - request_start_time_seconds) * 1000

    return jsonify({
        "text": recognized_text,
        "num_chars": len(recognized_text),
        "noise_profile": noise_profile,
        "latency_ms": round(latency_milliseconds, 2),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
