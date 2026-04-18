# 2-Stage Neural Compression Pipeline

A two-stage pipeline that ingests a noisy scanned document image, extracts its text using a CNN-based OCR model, and compresses the output using adaptive Huffman encoding - delivered as two communicating microservices. 

---

## Pipeline Overview

```
Noisy Document Image
        ↓
┌────────────────────────────────────────────┐
│          Stage 1: OCR Microservice          │  port 5001
│                                             │
│  1. DnCNN Denoiser                          │
│     noisy image → predict noise residual    │
│     → subtract → clean image                │
│     (skipped if output std < 0.05)          │
│                                             │
│  2. Character Segmentation                  │
│     Otsu binarize → connected components    │
│     → size filter → reading order sort      │
│     → list of 28×28 character crops         │
│                                             │
│  3. OCRNet CNN                              │
│     all crops → single batched fwd pass     │
│     → argmax over 62 classes → text string  │
└───────────────────┬────────────────────────┘
                    │  "Hello World 1234"
                    ↓
┌────────────────────────────────────────────┐
│       Stage 2: Compression Microservice     │  port 5002
│                                             │
│  POST /compress                             │
│    text → Adaptive Huffman encode           │
│    → compressed bytes + metrics             │
│                                             │
│  POST /decompress                           │
│    compressed bytes → decode                │
│    → original text (lossless ✓)             │
└────────────────────────────────────────────┘
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
│   ├── evaluate.py           # Pipeline accuracy on SimulatedNoisyOffice VA
│   ├── ocr_run.py            # Run OCR directly on an image (no server)
│   ├── server.py             # Flask API: POST /ocr, GET /health
│   └── weights/
│       ├── ocrnet.pth
│       ├── denoiser.pth
│       ├── metrics.json
│       └── denoiser_metrics.json
├── compression_service/
│   ├── huffman.py            # Adaptive Huffman (Vitter 1987) — from scratch
│   └── server.py             # Flask API: POST /compress, /decompress, GET /health
├── pipeline.py               # End-to-end orchestrator + latency report
├── data/
│   ├── English/Fnt/          # Chars74K — 62 classes, ~63k computer-font images
│   ├── SimulatedNoisyOffice/ # Digitally noisy documents (TR/VA/TE splits)
│   │   ├── simulated_noisy_images_grayscale/
│   │   └── clean_images_grayscale/
│   └── RealNoisyOffice/      # Real physically noisy scanned documents
│       └── real_noisy_images_grayscale/
└── requirements.txt
```

---

## Datasets

| Dataset | Purpose | Details |
|---|---|---|
| Chars74K Fnt | OCR training + validation | ~63k images, 62 classes, computer-rendered fonts |
| SimulatedNoisyOffice | Denoiser training, noise patch augmentation | Digital noise: creases, fade, warp, pepper |
| RealNoisyOffice | Noise patch augmentation | Real physical ink/water stains, high-res scans |

**Why Chars74K (Fnt split) used:**
The Chars74K dataset contains ~74,000 character images across 62 classes (digits 0–9, uppercase A–Z, lowercase a–z). The `Fnt` split specifically contains computer-rendered characters and each class has ~1,000 images generated from real system fonts at varying sizes, weights, and styles (serif, sans-serif, italic, bold). Images are grayscale on a white background, 28×28 after resizing. 

This matches the printed, computer-typeset text in the NoisyOffice dataset provided.

**How SimulatedNoisyOffice used:**
The dataset contains grayscale document page images (540×258 px) with printed text rendered in 18 different fonts. Each document appears in both a clean version and 4 noise variants:

| Noise code | Type |
|---|---|
| Noisec | Coffee / ink stains |
| Noisef | Fold marks and fade |
| Noisew | Wrinkle and warp distortion |
| Noisep | Footprint / pepper spots |

Images are split into three partitions — TR (train), VA (validation), TE (test) — giving 216 noisy images and 54 clean images total. Filenames encode all metadata: `Fontfre_Noisec_TR.png` means font `fre`, coffee-stain noise, training partition.

**How it's used to train the denoiser:**
The denoiser is trained on paired (noisy, clean) images from the TR partition. For each training sample, the noisy image is the input and the corresponding clean image (same font, same partition, e.g. `Fontfre_Clean_TR.png`) is the target. The model learns to predict the noise residual (input - clean), which is subtracted at inference to recover a clean image. The VA partition is used to track validation loss each epoch and save the best weights. The TE partition is held out for final evaluation.

**How it's used for OCR training (noise patch augmentation):**
During OCR training, random 28×28 patches are sampled from the TR noisy images and used as document-texture backgrounds. Each character crop is blended onto one of these patches - ink pixels kept dark, background replaced with real document texture - so the OCR model sees the same noise distribution as the evaluation documents.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

### Step 1 — Train OCR model

```bash
cd ocr_service
nohup python3 train.py --epochs 50 --batch-size 128 > nohup_train.log
tail -f nohup_train.log
```

**What it does:**
- Loads ~63k labeled character images from `../data/English/Fnt/` (62 classes)
- 85% train / 15% val random split (seed=42)
- Online noise augmentation on train split per sample:
  - 40% → real document patch background (SimulatedNoisyOffice)
  - 25% → Gaussian noise (random std 0.05–0.25)
  - 25% → salt-and-pepper noise (random prob 0.02–0.10)
  - 10% → clean (no augmentation)
- Adam optimiser (lr=5e-4, weight_decay=1e-4), cosine LR annealing
- Saves best weights (by clean val accuracy) → `weights/ocrnet.pth`
- Logs per-epoch: loss, clean accuracy, gaussian accuracy, salt-and-pepper accuracy → `weights/metrics.json`


### Step 2 — Train denoiser

```bash
cd ocr_service
nohup python3 train_denoiser.py --data-dir ../data/SimulatedNoisyOffice --epochs 50 > nohup_denoiser.log 2>&1 & echo "PID: $!"
```

**What it does:**
- Pairs noisy/clean document images by filename: `Fontfre_Noisec_TR.png` → `Fontfre_Clean_TR.png`
- MSE loss between denoiser output and clean image (residual learning)
- Validates on VA partition every epoch
- Saves best weights (by val loss) → `weights/denoiser.pth`

### Step 3 — Evaluate pipeline accuracy

```bash
cd ocr_service
python3 evaluate.py --partition VA
```

**What it does:**
- For each font in the VA partition, runs the full pipeline on the clean image → ground truth text
- Runs the same pipeline on each noisy variant → predicted text
- Reports character-level accuracy per noise type and overall
- Target: ≥ 95%

---

## Running the Full Pipeline

### Start both microservices

```bash
# Terminal 1
cd ocr_service
python3 server.py

# Terminal 2
cd compression_service
python3 server.py
```

Health check:
```bash
curl http://localhost:5001/health
curl http://localhost:5002/health
```

### End-to-end pipeline

```bash
python3 pipeline.py --image data/SimulatedNoisyOffice/simulated_noisy_images_grayscale/Fontfre_Noisec_TE.png
```

With noise profile benchmarking:
```bash
python3 pipeline.py --image path/to/image.png --noise-profile gaussian
python3 pipeline.py --image path/to/image.png --noise-profile salt_and_pepper
python3 pipeline.py --image path/to/image.png --noise-profile sidd
```

### OCR only (no server needed)

```bash
cd ocr_service
python3 ocr_run.py --image ../data/SimulatedNoisyOffice/simulated_noisy_images_grayscale/Fontfre_Noisec_TE.png
```

### Direct API calls

```bash
# OCR
curl -X POST http://localhost:5001/ocr -F "image=@image.png"

# OCR with noise profile applied before processing
curl -X POST http://localhost:5001/ocr \
  -F "image=@image.png" \
  -F "noise_profile=gaussian"

# Compress
curl -X POST http://localhost:5002/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello World 1234"}'

# Decompress
curl -X POST http://localhost:5002/decompress \
  -H "Content-Type: application/json" \
  -d '{"data": "<hex_string>", "bit_length": <N>}'
```

---

## Stage 1 — OCR Microservice

### OCRNet Architecture

#### Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: 1 × 28 × 28                        │
│                  (grayscale character crop)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │         BLOCK 1             │
              │  Conv2d(1→32, 3×3, pad=1)   │
              │  BatchNorm2d(32)            │
              │  ReLU                       │
              │  Conv2d(32→32, 3×3, pad=1)  │
              │  BatchNorm2d(32)            │
              │  ReLU                       │
              │  MaxPool2d(2×2)  28→14      │
              │  Dropout2d(p=0.15)          │
              │  Output: 32 × 14 × 14       │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │         BLOCK 2             │
              │  Conv2d(32→64, 3×3, pad=1)  │
              │  BatchNorm2d(64)            │
              │  ReLU                       │
              │  Conv2d(64→64, 3×3, pad=1)  │
              │  BatchNorm2d(64)            │
              │  ReLU                       │
              │  MaxPool2d(2×2)  14→7       │
              │  Dropout2d(p=0.15)          │
              │  Output: 64 × 7 × 7         │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼───────────────┐
              │          BLOCK 3             │
              │  Conv2d(64→128, 3×3, pad=1)  │
              │  BatchNorm2d(128)            │
              │  ReLU                        │
              │  Conv2d(128→128, 3×3, pad=1) │
              │  BatchNorm2d(128)            │
              │  ReLU                        │
              │  MaxPool2d(2×2)  7→3         │
              │  Dropout2d(p=0.15)           │
              │  Output: 128 × 3 × 3         │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼──────────────┐
              │        CLASSIFIER           │
              │  Flatten → 1152             │
              │  Linear(1152 → 512)         │
              │  ReLU                       │
              │  Dropout(p=0.4)             │
              │  Linear(512 → 256)          │
              │  ReLU                       │
              │  Dropout(p=0.3)             │
              │  Linear(256 → 62)           │
              │  LogSoftmax(dim=1)          │
              └─────────────┬──────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              OUTPUT: log-probabilities over 62 classes        │
│         0–9 (digits), A–Z (uppercase), a–z (lowercase)       │
└─────────────────────────────────────────────────────────────┘
```

**Total trainable parameters:** ~2.1M

#### Design Justifications

| Design Choice | Justification |
|---|---|
| **Input size: 28×28** | Standard for character recognition; each segmented crop is resized to this before classification |
| **3×3 kernels throughout** | Optimal accuracy-to-parameter ratio for small images; two 3×3 layers capture the same receptive field as one 5×5 with fewer parameters and an extra non-linearity |
| **Filter depth 32→64→128** | Progressive feature hierarchy — early layers detect simple edges and strokes; deeper layers combine these into full character structure |
| **2 conv layers per block** | Doubles representational capacity per spatial scale vs a single conv; all three blocks are symmetric for consistent depth |
| **BatchNorm after every conv** | Normalises activations per mini-batch — stabilises training under variable noise levels and accelerates convergence by reducing internal covariate shift |
| **MaxPool2d(2×2)** | Halves spatial dimensions at each block; provides translation invariance to small crop misalignments from segmentation |
| **Dropout2d(0.15) in conv blocks** | Drops entire feature maps rather than individual neurons — stronger spatial regularisation; kept at 0.15 (not 0.25) because noise augmentation already acts as regularisation, and higher values caused underfitting |
| **FC layers: 512→256→62** | Two hidden layers needed for 62-class complexity; a single 1152→62 projection underfit; 512→256 progressively compresses features before classification |
| **Dropout 0.4/0.3 in FC** | Prevents co-adaptation of FC neurons; lower than typical (0.5) to avoid underfitting given heavy input augmentation |
| **LogSoftmax + NLLLoss** | Numerically equivalent to CrossEntropyLoss but allows monitoring raw log-probabilities separately from the loss computation |
| **ReLU activations** | Computationally efficient; avoids vanishing gradient problem in deeper networks vs sigmoid/tanh |

**Validation accuracy (Chars74K 15% hold-out, best model at epoch 45/50):**

| Noise profile | Accuracy |
|---|---|
| Clean | 89.50% |
| Gaussian | 88.77% |
| Salt-and-pepper | 89.24% |

Training loss dropped from 2.11 → 0.35 over 50 epochs. Best weights saved at epoch 45 (by clean val accuracy).

### Character Segmentation

`segment.py` — fully from-scratch connected-component segmentation; no OCR libraries used.

**Steps:**

1. **Grayscale conversion** — input PIL Image converted to NumPy float32 array
2. **Otsu thresholding (from scratch)** — sweeps all 256 intensity levels and picks the threshold that maximises between-class variance (background pixels vs ink pixels). Fully implemented without `cv2` or `skimage` — just NumPy histograms. Adaptive to any contrast level, so the same code works on a high-contrast printed page and a faded, stained scan.
3. **Binarize** — if the image has a light background (mean > 127), ink pixels are those below the threshold. If the background is dark (inverted scan), the logic is flipped. This handles both document orientations automatically.
4. **Connected component labeling** — `scipy.ndimage.label()` groups all touching ink pixels into labelled blobs. Each blob is a candidate character.
5. **Size and shape filtering** — thresholds scale with image resolution so the same code works on 28×28 crops and full 540×420 page scans:
   - `min_area = max(20, 0.015% of image pixels)` — removes isolated noise specks too small to be a character
   - `max_area = 2% of image pixels` — removes large ink stains, coffee marks, and smudges that cover too much area to be a single character
   - `min_width = max(3, 0.3% of image width)` — removes thin vertical streaks
   - Aspect ratio: blobs wider than 8× their height are discarded as horizontal rules or underlines
6. **Reading order sort** — bounding boxes are grouped into text lines using the median character height as a vertical tolerance band. Within each line, boxes are sorted left-to-right. Lines are sorted top-to-bottom. This produces correct reading order without any ML model.
7. **Crop and resize** — each accepted bounding box is extracted as a grayscale PIL Image and passed to the OCRNet (resized to 28×28 inside the CNN preprocessing transform).

**Why connected components instead of a sliding window or YOLO-style detector:**
Connected components are fast (< 50 ms on a full page), require no training data, and naturally handle variable character sizes and spacings. The size filters are the only tunable parameters, and they scale automatically with resolution.

---

### DnCNN Denoiser

`denoiser.py` — residual noise learning; fully convolutional so it handles any image resolution without resizing.

**Architecture:**

```
Input: grayscale document image, shape (1, H, W), values in [0, 1]

Conv2d(1 → 64, kernel=3×3, pad=1) → ReLU          # feature extraction, no BN
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    ┐
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    │
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    │  × 8 middle layers
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    │  (total 10 layers)
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    │
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    │
Conv2d(64 → 64, kernel=3×3, pad=1) → BN → ReLU    ┘
Conv2d(64 → 1,  kernel=3×3, pad=1)                 # noise residual map

Output = clamp(Input − predicted_noise, 0.0, 1.0)  # clean image
```

**Why residual learning:** The network predicts the noise pattern (what to remove), not the clean image (what to keep). Noise is easier to learn — it is spatially local and low-amplitude. Predicting the clean image directly forces the network to reconstruct fine text strokes exactly, which is much harder. Subtracting the predicted noise from the input acts as a skip connection, similar to ResNets.

**Training:** Paired (noisy, clean) SimulatedNoisyOffice images from the TR partition. Loss is MSE between `clamp(noisy − predicted_noise, 0, 1)` and the clean target. 4 noise types × 18 fonts = 72 training pairs. VA partition used for early stopping.

**Fallback behaviour:** After denoising, if the output image has standard deviation < 0.05 (indicating the model collapsed contrast and destroyed the document), the pipeline automatically falls back to the original undenoised image. This prevents a bad denoiser pass from breaking the downstream segmentation step.

**Noise profiles handled:**

| Code | Type | Visual effect |
|---|---|---|
| Noisec | Coffee / ink stains | Dark irregular blobs on background |
| Noisef | Fold marks and fade | Diagonal bright/dark bands across page |
| Noisew | Wrinkle and warp | Local geometric distortion + brightness variation |
| Noisep | Footprint / pepper spots | Small dark speckles scattered across page |

---

## Stage 2 — Compression Microservice

### Adaptive Huffman Encoding

`compression_service/huffman.py` — Vitter's 1987 algorithm, implemented entirely from scratch. No compression libraries used.

Unlike standard Huffman, no frequency table is transmitted — the encoder and decoder both build the same tree symbol-by-symbol as they process the data.

**Algorithm key concepts:**

- **NYT node** (Not Yet Transmitted): placeholder for unseen symbols; emits NYT path + 8-bit ASCII when a new character is first seen
- **Sibling property**: nodes ordered by weight (non-decreasing left to right); maintained after every symbol via slide-and-increment
- **Block index** (`weight → list[Node]`): O(1) lookup for same-weight nodes; replaces O(n) DFS used in naive implementations
- **Swap rule**: when a node's weight should increase, first swap it with the highest-order node of the same weight (if not ancestor/descendant), then increment — this maintains the sibling property

**Encoding example — `"hello"`:**
```
'h' → new symbol: emit NYT_code + 8-bit ASCII('h'); add 'h' to tree
'e' → new symbol: emit NYT_code + 8-bit ASCII('e'); add 'e' to tree
'l' → new symbol: emit NYT_code + 8-bit ASCII('l'); add 'l' to tree
'l' → seen:       emit Huffman code for 'l' (now shorter); l.weight → 2
'o' → new symbol: emit NYT_code + 8-bit ASCII('o'); add 'o' to tree
```

After processing, 'l' has the shortest code since it appeared most. The tree self-organises with no pre-scan of the data.

### Compression Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Compression ratio | `original_bits / compressed_bits` | >1.0 means file is smaller |
| Entropy | `−Σ p(c) log₂ p(c)` | Theoretical minimum bits per symbol (Shannon) |
| Avg bits/symbol | `compressed_bits / num_chars` | Actual bits used per character |
| Encoding efficiency | `entropy / avg_bits_per_symbol` | 1.0 = theoretically optimal |

### API

**POST /compress**
```json
Request:  {"text": "Hello World"}
Response: {
  "data": "a3f04c...",
  "bit_length": 89,
  "metrics": {
    "original_bytes": 11,
    "compressed_bytes": 12,
    "compression_ratio": 0.733,
    "entropy_bits_per_symbol": 3.095,
    "avg_bits_per_symbol": 8.0,
    "encoding_efficiency": 0.387
  },
  "latency_ms": 0.04
}
```

**POST /decompress**
```json
Request:  {"data": "a3f04c...", "bit_length": 89}
Response: {"text": "Hello World", "latency_ms": 0.03}
```

**GET /health**
```json
{"status": "ok"}
```

---

## End-to-End Latency

| Stage | Component | Latency |
|---|---|---|
| Stage 1 | DnCNN denoiser | ~200 ms |
| Stage 1 | Segmentation (Otsu + connected components) | ~50 ms |
| Stage 1 | OCRNet inference (batched) | ~350 ms |
| **Stage 1 total** | | **~600 ms** |
| Stage 2 | Compress | ~0.04 ms |
| Stage 2 | Decompress | ~0.03 ms |
| **End-to-end** | | **~620 ms** |

---

## Graduate Requirements Checklist

- [x] CNN trained from scratch using PyTorch
- [x] OCR microservice exposes `POST /ocr` endpoint accepting image, returning text
- [x] Adaptive Huffman implemented entirely from scratch (no zlib/gzip)
- [x] Compression microservice exposes `POST /compress` and `POST /decompress`
- [x] Lossless decompression verified (`recovered == original` assert in `pipeline.py`)
- [x] Two noise profiles: Gaussian + salt-and-pepper, accuracy logged per epoch
- [x] SIDD noise profile supported (`noise.py`) for real-world evaluation
- [x] Compression ratio, entropy, and encoding efficiency reported per request
- [x] End-to-end latency benchmarked and reported above
- [x] CNN architecture documented with diagram and design justification table
- [x] Pipeline evaluation script (`evaluate.py`) measuring character accuracy on VA partition
- [x] DnCNN denoiser trained on SimulatedNoisyOffice noisy/clean pairs
- [x] Document noise augmentation using real NoisyOffice patches during OCR training
- [ ] Fill accuracy table after training completes
- [ ] Record demo video (image in → compressed output → decompressed text)

---

## Team

- Pradyot Bathuri
- Michelle Benites Mendez
- Anooshka Bajaj