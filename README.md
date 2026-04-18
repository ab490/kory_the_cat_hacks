# OCR + Adaptive Huffman (two services)

Stage 1 (port 5001): PyTorch CNN reads text from an image. Stage 2 (port 5002): adaptive Huffman compresses that text; decompress gets it back losslessly. `pipeline.py` calls both.

## Setup

Use Python 3 with a venv, then install deps:

`pip install -r requirements.txt`

Put `ocrnet.pth` and `denoiser.pth` in `ocr_service/weights/` (both required for `server.py`). Train with the commands below if you need to regenerate them.

## Run the pipeline

You need two terminals for the servers, then one for the client.

1. OCR service: `cd ocr_service` then `python3 server.py` (listens on **5001**; override with `PORT=5003` if busy).

2. Compression service: `cd compression_service` then `python3 server.py` (listens on **5002**; override with `PORT=` same way).

3. From the repo root: `python3 pipeline.py --image path/to/your.png`

You should see extracted text, compression metrics, `Lossless check: PASS`, and total time in ms.

Optional: `python3 pipeline.py --image test_alpha.png --noise-profile gaussian` to add noise before OCR (also `salt_and_pepper`, `sidd`). If servers use other ports: `--ocr-url http://localhost:5003 --compression-url http://localhost:5004`.

## Train or retrain models

OCR (downloads EMNIST into `../data`, saves `ocr_service/weights/ocrnet.pth` and `metrics.json`):

`cd ocr_service && python3 train.py --epochs 25`

Training also pulls TrueType fonts from Hugging Face (`jonathang/fonts-ttf`); the first run may download them into the Hugging Face cache.

Denoiser (needs `SimulatedNoisyOffice` next to or under the repo; saves `denoiser.pth`):

`cd ocr_service && python3 train_denoiser.py --data-dir ../SimulatedNoisyOffice --epochs 50`

## What’s where

- `ocr_service/` — CNN, denoiser, segmentation, Flask `server.py`, `train.py`, `train_denoiser.py`
- `compression_service/` — `huffman.py` (encoder/decoder), Flask `server.py`
- `pipeline.py` — end-to-end client
- `benchmark.py` — OCR-only benchmark against NoisyOffice images (server must be up)

## Demo for judges

Start both servers, run `pipeline.py` on a sample image, show the printed output and that lossless recovery passes.
