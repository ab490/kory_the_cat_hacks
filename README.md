# 2-Stage Neural Compression Pipeline

A two-stage pipeline that ingests a noisy scanned document image, extracts its text using a CNN-based OCR model, and compresses the output using adaptive Huffman encoding - delivered as two communicating microservices.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Training](#training)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Stage 1 - OCR Microservice](#stage-1---ocr-microservice)
  - [OCRNet Architecture](#ocrnet-architecture)
  - [Design Justifications](#design-justifications)
  - [Character Segmentation](#character-segmentation)
  - [DnCNN Denoiser](#dncnn-denoiser)
- [Stage 2 - Compression Microservice](#stage-2---compression-microservice)
  - [Compression Metrics](#compression-metrics)
- [End-to-End Latency](#end-to-end-latency)
- [Requirements Checklist](#requirements-checklist)
- [Team](#team)

---

## Pipeline Overview

```
Noisy Document Image
        ↓
[Stage 1: OCR Microservice - port 5001]
  DnCNN Denoiser -> Character Segmentation -> OCRNet CNN
        ↓
  "Extracted Text"
        ↓
[Stage 2: Compression Microservice - port 5002]
  POST /compress  -> Adaptive Huffman encode -> compressed bytes + metrics
  POST /decompress -> decode -> original text (lossless check)
```

---

## Repository Structure

```
├── ocr_service/
│   ├── model.py              # OCRNet CNN (3 conv blocks, 62 classes)
│   ├── denoiser.py           # DnCNN denoiser (10-layer residual)
│   ├── noise.py              # Gaussian, salt-and-pepper, SIDD noise functions
│   ├── segment.py            # Connected-component character segmentation
│   ├── train.py              # Train OCRNet on Chars74K + document noise
│   ├── train_denoiser.py     # Train denoiser on SimulatedNoisyOffice pairs
│   ├── server.py             # Flask API: POST /ocr, GET /health
│   └── weights/
│       ├── ocrnet.pth
│       ├── denoiser.pth
│       ├── metrics.json
│       └── denoiser_metrics.json
├── compression_service/
│   ├── huffman.py            # Adaptive Huffman
│   └── server.py             # Flask API: POST /compress, /decompress, GET /health
├── data/                     # not in repo - download separately (see Setup)
│   ├── Chars74K/Fnt/         
│   ├── SimulatedNoisyOffice/ 
│   └── RealNoisyOffice/      
├── results/
│   ├── pipeline_result.txt   # Sample end-to-end pipeline run output
│   ├── nohup_train.log       # OCR model training log
│   └── nohup_denoiser.log    # Denoiser training log
├── pipeline.py               # End-to-end orchestrator
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Data:** Download the three dataset zips from One Drive: [**datasets**](https://indiana-my.sharepoint.com/:f:/g/personal/anobajaj_iu_edu/IgCKXF0AFaLMTZMoTI18_dV3AQ_-3rpB_uQiw9D9OjIo89I?e=4Nv1Vz)

```bash
mkdir -p data
unzip Chars74K.zip -d data/
unzip SimulatedNoisyOffice.zip -d data/
unzip RealNoisyOffice.zip -d data/
```

---

## Training

```bash
# Step 1 - Train OCR model (~50 epochs)
cd ocr_service
python3 train.py --epochs 50 --batch-size 128

# Step 2 - Train denoiser (~50 epochs)
python3 train_denoiser.py --data-dir ../data/SimulatedNoisyOffice --epochs 50
```

Pre-trained weights are included in `ocr_service/weights/` - training is only needed to reproduce from scratch.

---

## Running the Full Pipeline

### Start both microservices

```bash
# Terminal 1
cd ocr_service && python3 server.py

# Terminal 2
cd compression_service && python3 server.py
```

### End-to-end pipeline

```bash
python3 pipeline.py --image <path_to_image>
```

---

## Stage 1 - OCR Microservice

### OCRNet Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: 1 × 28 × 28                       │
│                  (grayscale character crop)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │         BLOCK 1            │
              │  Conv2d(1->32, 3×3, pad=1) │
              │  BatchNorm2d(32)           │
              │  ReLU                      │
              │  Conv2d(32->32, 3×3, pad=1)│
              │  BatchNorm2d(32)           │
              │  ReLU                      │
              │  MaxPool2d(2×2)  28->14    │
              │  Dropout2d(p=0.15)         │
              │  Output: 32 × 14 × 14      │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │         BLOCK 2            │
              │  Conv2d(32->64, 3×3, pad=1)│
              │  BatchNorm2d(64)           │
              │  ReLU                      │
              │  Conv2d(64->64, 3×3, pad=1)│
              │  BatchNorm2d(64)           │
              │  ReLU                      │
              │  MaxPool2d(2×2)  14->7     │
              │  Dropout2d(p=0.15)         │
              │  Output: 64 × 7 × 7        │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼───────────────┐
              │          BLOCK 3            │
              │ Conv2d(64->128, 3×3, pad=1) │
              │ BatchNorm2d(128)            │
              │ ReLU                        │
              │ Conv2d(128->128, 3×3, pad=1)│
              │ BatchNorm2d(128)            │
              │ ReLU                        │
              │ MaxPool2d(2×2)  7->3        │
              │ Dropout2d(p=0.15)           │
              │ Output: 128 × 3 × 3         │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼──────────────┐
              │        CLASSIFIER          │
              │  Flatten -> 1152           │
              │  Linear(1152 -> 512)       │
              │  ReLU                      │
              │  Dropout(p=0.4)            │
              │  Linear(512 -> 256)        │
              │  ReLU                      │
              │  Dropout(p=0.3)            │
              │  Linear(256 -> 62)         │
              │  LogSoftmax(dim=1)         │
              └─────────────┬──────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              OUTPUT: log-probabilities over 62 classes      │
│         0–9 (digits), A–Z (uppercase), a–z (lowercase)      │
└─────────────────────────────────────────────────────────────┘
```

**Total trainable parameters:** ~2.1M

### Design Justifications

| Design Choice | Justification |
|---|---|
| **Input size: 28×28** | Standard for character recognition; each segmented crop is resized before classification |
| **3×3 kernels throughout** | Optimal accuracy-to-parameter ratio; two 3×3 layers capture the same receptive field as one 5×5 with fewer parameters and an extra non-linearity |
| **Filter depth 32->64->128** | Progressive feature hierarchy - early layers detect edges and strokes; deeper layers combine into full character structure |
| **2 conv layers per block** | Doubles representational capacity per spatial scale vs a single conv |
| **BatchNorm after every conv** | Stabilises training under variable noise levels, accelerates convergence |
| **MaxPool2d(2×2)** | Halves spatial dimensions at each block; provides translation invariance to segmentation misalignments |
| **Dropout2d(0.15) in conv blocks** | Drops entire feature maps - stronger spatial regularisation; kept low because noise augmentation already regularises |
| **FC layers: 512->256->62** | Single 1152->62 projection underfit; two hidden layers needed for 62-class complexity |
| **LogSoftmax + NLLLoss** | Equivalent to CrossEntropyLoss but separates log-probability monitoring from loss |
| **ReLU activations** | Avoids vanishing gradient vs sigmoid/tanh; computationally efficient |


### Character Segmentation

`segment.py`: fully from-scratch connected-component segmentation.

Otsu threshold (from scratch) -> binarize -> `scipy.ndimage.label()` -> resolution-adaptive size filter -> reading-order sort -> 28×28 crops. All thresholds scale with image resolution so the same code handles 28×28 crops and full 540×420 page scans.

### DnCNN Denoiser

`denoiser.py` - residual noise learning; fully convolutional, handles any image resolution.

```
Input: (1, H, W) grayscale, values in [0, 1]

Conv(1->64, 3×3) -> ReLU
× 8: Conv(64->64, 3×3) -> BN -> ReLU
Conv(64->1, 3×3)

Output = clamp(Input - predicted_noise, 0.0, 1.0)
```

Trained on paired (noisy, clean) SimulatedNoisyOffice TR images. If the denoised output has std < 0.05 (contrast collapse), the pipeline falls back to the original image.

---

## Stage 2 - Compression Microservice

`compression_service/huffman.py` - Vitter-style adaptive Huffman from scratch. Encoder and decoder build the same tree symbol-by-symbol as data is processed - no frequency pre-scan, no transmitted frequency table. New symbols emit NYT (not-yet-transmitted) path + 8-bit ASCII; seen symbols emit their current Huffman code.

### Compression Metrics Results

| Metric | Value |
|---|---|
| Original bytes | 478 |
| Compressed bytes | 1073 |
| Compression ratio | 0.4456 |
| Entropy (bits/sym) | 4.6561 |
| Avg bits/symbol | 17.9519 |
| Encoding efficiency | 0.2594 |

---

## End-to-End Latency

| Stage | Component | Latency |
|---|---|---|
| Stage 1 | DnCNN denoiser | ~200 ms |
| Stage 1 | Segmentation (Otsu + connected components) | ~50 ms |
| Stage 1 | OCRNet inference (batched) | ~373 ms |
| **Stage 1 total** | | **622.88 ms** |
| Stage 2 | Compress | 39.22 ms |
| Stage 2 | Decompress | 39.30 ms |
| **End-to-end** | | **713.21 ms** |

---

## Requirements Checklist

### Stage 1 - OCR Microservice (CNN)
- [x] CNN built and trained from scratch using PyTorch (no pretrained models)
- [x] `POST /ocr` endpoint accepts an image and returns extracted text
- [x] Two noise profiles supported with measurable accuracy on each:

| Noise profile | Chars74K val accuracy |
|---|---|
| Clean | 89.50% |
| Gaussian | 88.77% |
| Salt-and-pepper | 89.24% |

- [x] CNN architecture documented with diagram and design justification - see [OCRNet Architecture](#ocrnet-architecture)

### Stage 2 - Compression Microservice (Adaptive Huffman)
- [x] Adaptive Huffman implemented entirely from scratch
- [x] `POST /compress` and `POST /decompress` endpoints implemented
- [x] Lossless decompression verified (`recovered == original` assert in `pipeline.py`)
- [x] Compression ratio, entropy, and encoding efficiency reported per `/compress` response - see [Compression Metrics](#compression-metrics)

### Pipeline
- [x] End-to-end latency benchmarked - see [End-to-End Latency](#end-to-end-latency) (~713 ms total)

---

## Team

- Pradyot Bathuri
- Michelle Benites Mendez
- Anooshka Bajaj
