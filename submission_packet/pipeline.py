# End-to-end demo: image -> OCR -> adaptive Huffman compress -> decompress -> verify.
#
# Start both servers first, then run:
#   python pipeline.py --image path/to/test.png

import argparse
import json
import time

import requests

DEFAULT_OCR = "http://localhost:5001"
DEFAULT_COMPRESS = "http://localhost:5002"


def run(image_path, ocr_url, compress_url):
    t0 = time.time()

    print(f"[1] OCR @ {ocr_url}")
    with open(image_path, "rb") as f:
        r = requests.post(f"{ocr_url}/ocr", files={"image": f}, timeout=300)
    r.raise_for_status()
    ocr = r.json()
    text = ocr["text"]
    print(f"    text ({len(text)} chars):\n{text}")
    print(f"    ocr latency: {ocr['latency_ms']} ms")

    if not text:
        raise SystemExit("OCR returned empty text; nothing to compress.")

    print(f"\n[2] Compress @ {compress_url}")
    r = requests.post(f"{compress_url}/compress", json={"text": text}, timeout=60)
    r.raise_for_status()
    comp = r.json()
    m = comp["metrics"]
    print(f"    {m['original_bytes']}B -> {m['compressed_bytes']}B  "
          f"(ratio {m['compression_ratio']}, "
          f"avg {m['avg_bits_per_symbol']} bits/sym, "
          f"entropy {m['entropy_bits_per_symbol']})")

    print(f"\n[3] Decompress @ {compress_url}")
    r = requests.post(f"{compress_url}/decompress",
                      json={"data": comp["data"], "bit_length": comp["bit_length"]},
                      timeout=60)
    r.raise_for_status()
    recovered = r.json()["text"]
    ok = recovered == text
    print(f"    lossless: {'PASS' if ok else 'FAIL'}")

    total = round((time.time() - t0) * 1000, 2)
    print(f"\ntotal: {total} ms")
    return {"text": text, "recovered": recovered, "lossless": ok,
            "metrics": m, "total_ms": total}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ocr-url", default=DEFAULT_OCR)
    ap.add_argument("--compression-url", default=DEFAULT_COMPRESS)
    args = ap.parse_args()
    result = run(args.image, args.ocr_url.rstrip("/"), args.compression_url.rstrip("/"))
    print("\n" + json.dumps(result, indent=2))
