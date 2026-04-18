import io
import os
import time

import torch
from flask import Flask, jsonify, request
from PIL import Image

from infer import detect_boxes, load_models, recognize

WEIGHTS = os.path.join(os.path.dirname(__file__), "weights")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
print(f"[ocr] loading models on {DEV}")
DETECTOR, RECOGNIZER = load_models(
    os.path.join(WEIGHTS, "detector.pt"),
    os.path.join(WEIGHTS, "recognizer.pt"),
    DEV,
)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEV)})


@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "upload field 'image' is required"}), 400
    start = time.time()
    pil = Image.open(io.BytesIO(request.files["image"].read())).convert("L")
    boxes = detect_boxes(DETECTOR, pil, DEV) or [(0, 0, pil.width, pil.height)]
    lines = [recognize(RECOGNIZER, pil.crop(b), DEV) for b in boxes]
    return jsonify({
        "text": "\n".join(lines),
        "boxes": [list(b) for b in boxes],
        "latency_ms": round((time.time() - start) * 1000, 2),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
