# OCR + Adaptive Huffman

Two tiny microservices wired together:

- **Stage 1 — OCR (`ocr_service/`, port 5001)**: a compact PyTorch pipeline.
  `DBNet` predicts which pixels contain text, and a `CRNN + CTC` recognizer
  reads each line crop.
- **Stage 2 — Compression (`compression_service/`, port 5002)**: a
  from-scratch adaptive Huffman coder (FGK / Vitter) that losslessly
  compresses the OCR output.

`pipeline.py` is an end-to-end client: image in → compressed bytes out →
decompressed text recovered → lossless check.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the models

We train on a tiny synthetic dataset of rendered paragraphs; 10-15 minutes on
a laptop CPU is plenty for the detector, another 20-30 for the recognizer.

```bash
cd ocr_service

# 1. generate a few hundred synthetic pages (+ line crops) for training
python synth.py --output_dir data --num_pages 400

# 2. train the DBNet detector on the full pages
python train.py --stage detector \
    --manifest data/pages_manifest.json --data_root data \
    --output_dir weights --epochs 20

# 3. train the CRNN recognizer on the line crops
python train.py --stage recognizer \
    --manifest data/lines_manifest.json --data_root data \
    --output_dir weights --epochs 30
```

The final checkpoints land at `ocr_service/weights/detector.pt` and
`ocr_service/weights/recognizer.pt`. Those are the two files `server.py`
loads at startup.

## Run the pipeline

Open three terminals:

```bash
# terminal 1 — OCR service
cd ocr_service && python server.py          # :5001

# terminal 2 — compression service
cd compression_service && python server.py   # :5002

# terminal 3 — client
python pipeline.py --image path/to/page.png
```

You should see the OCR output, the Huffman metrics (original bytes,
compressed bytes, ratio, avg bits/symbol), and a final `lossless: PASS`.

Flag overrides:

```bash
python pipeline.py --image test.png \
    --ocr-url http://localhost:5003 \
    --compression-url http://localhost:5004
```

## Layout

```
submission_packet/
├── ocr_service/
│   ├── config.py         # charset + image sizes
│   ├── model.py          # DBNet + CRNN
│   ├── decode.py         # greedy CTC decode
│   ├── transforms.py     # tensor helpers
│   ├── synth.py          # tiny synthetic data generator
│   ├── data.py           # two Dataset classes
│   ├── train.py          # trains detector or recognizer
│   ├── infer.py          # detect + recognize one image
│   ├── server.py         # Flask /ocr endpoint
│   └── weights/          # detector.pt, recognizer.pt land here
├── compression_service/
│   ├── huffman.py        # from-scratch adaptive Huffman (FGK)
│   └── server.py         # Flask /compress + /decompress
├── pipeline.py           # end-to-end demo client
├── benchmark.py          # OCR sweep over SimulatedNoisyOffice TE set
└── requirements.txt
```

## Benchmark (optional)

If you have the SimulatedNoisyOffice dataset dropped next to the repo, start
the OCR server and run:

```bash
python benchmark.py
```

This reads each font's clean TE page plus its four noisy variants and prints
the average character-match rate, so we can show how OCR degrades with
scanner noise.

## Judge demo (30 seconds)

1. Both servers are already running.
2. `python pipeline.py --image sample.png` — extracted text, compression
   ratio, `lossless: PASS`.
3. Point at the architecture: DBNet for detection, CRNN + CTC for
   recognition, adaptive Huffman for compression — the three pieces of the
   spec in under a thousand lines of code.
