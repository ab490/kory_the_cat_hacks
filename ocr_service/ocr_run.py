# Local OCR: python ocr_run.py --image file.png

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import OCRNet
from denoiser import Denoiser
from segment import segment_characters

LABEL_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "ocrnet.pth")
DENOISER_PATH = os.path.join(os.path.dirname(
    __file__), "weights", "denoiser.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isfile(WEIGHTS_PATH):
    sys.exit("missing weights/ocrnet.pth")
if not os.path.isfile(DENOISER_PATH):
    sys.exit("missing weights/denoiser.pth")

model = OCRNet(num_character_classes=47).to(device)
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


def run_ocr(image_path):
    source_image = Image.open(image_path)
    grayscale_batch = denoise_transform(source_image).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_normalized_array = denoiser(grayscale_batch).squeeze().cpu().numpy()
    denoised_image = Image.fromarray((denoised_normalized_array * 255).astype(np.uint8))

    character_crops = segment_characters(denoised_image)
    if len(character_crops) == 0:
        return ""
    character_batch_tensor = torch.stack([char_transform(crop) for crop in character_crops]).to(device)
    with torch.no_grad():
        predicted_character_class_indices = model(character_batch_tensor).argmax(dim=1).tolist()
    return "".join(LABEL_MAP[class_index] for class_index in predicted_character_class_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(run_ocr(args.image))
