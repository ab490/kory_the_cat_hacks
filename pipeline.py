"""
End-to-end pipeline orchestrator.
Sends an image to Stage 1 (OCR), then the text to Stage 2 (compress/decompress).
Reports full metrics and end-to-end latency.
"""

import argparse
import time
import requests
import json

OCR_URL = "http://localhost:5001"
COMPRESS_URL = "http://localhost:5002"


def run_pipeline(image_path: str, cols: int = 1, noise_profile: str = "none"):
    t_start = time.time()

    # Stage 1: OCR
    print(f"\n[Stage 1] Sending image to OCR service...")
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{OCR_URL}/ocr",
            files={"image": f},
            data={"cols": cols, "noise_profile": noise_profile},
        )
    resp.raise_for_status()
    ocr_result = resp.json()
    text = ocr_result["text"]
    print(f"  Extracted text   : {text!r}")
    print(f"  Noise profile    : {ocr_result['noise_profile']}")
    print(f"  OCR latency      : {ocr_result['latency_ms']} ms")

    # Stage 2: Compress
    print(f"\n[Stage 2] Sending text to compression service...")
    resp = requests.post(
        f"{COMPRESS_URL}/compress",
        json={"text": text},
    )
    resp.raise_for_status()
    compress_result = resp.json()
    print(f"  Compressed hex   : {compress_result['data'][:40]}{'...' if len(compress_result['data']) > 40 else ''}")
    print(f"  Compress latency : {compress_result['latency_ms']} ms")
    m = compress_result["metrics"]
    print(f"  Metrics:")
    print(f"    original bytes       : {m['original_bytes']}")
    print(f"    compressed bytes     : {m['compressed_bytes']}")
    print(f"    compression ratio    : {m['compression_ratio']}")
    print(f"    entropy (bits/sym)   : {m['entropy_bits_per_symbol']}")
    print(f"    avg bits/symbol      : {m['avg_bits_per_symbol']}")
    print(f"    encoding efficiency  : {m['encoding_efficiency']}")

    # Stage 2: Decompress
    print(f"\n[Stage 2] Decompressing...")
    resp = requests.post(
        f"{COMPRESS_URL}/decompress",
        json={"data": compress_result["data"], "bit_length": compress_result["bit_length"]},
    )
    resp.raise_for_status()
    decompress_result = resp.json()
    recovered_text = decompress_result["text"]
    print(f"  Recovered text   : {recovered_text!r}")
    print(f"  Decompress latency: {decompress_result['latency_ms']} ms")

    # Verify lossless
    match = recovered_text == text
    print(f"\n  Lossless check   : {'PASS' if match else 'FAIL'}")

    t_end = time.time()
    total_latency_ms = round((t_end - t_start) * 1000, 2)
    print(f"\n[Pipeline] End-to-end latency: {total_latency_ms} ms")

    return {
        "text": text,
        "recovered_text": recovered_text,
        "lossless": match,
        "metrics": m,
        "total_latency_ms": total_latency_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--cols", type=int, default=1, help="Number of digit columns in image")
    parser.add_argument("--noise-profile", default="none",
                        choices=["none", "gaussian", "salt_and_pepper", "sidd"])
    args = parser.parse_args()

    result = run_pipeline(args.image, cols=args.cols, noise_profile=args.noise_profile)
    print("\n" + json.dumps(result, indent=2))