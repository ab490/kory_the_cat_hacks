# Run after: ocr_service/server.py on 5001, compression_service/server.py on 5002
# python pipeline.py --image file.png

import argparse
import json
import time

import requests

OCR_URL = "http://localhost:5001"
COMPRESSION_URL = "http://localhost:5002"


def run_pipeline(
    image_path,
    cols=1,
    noise_profile="none",
    ocr_base=OCR_URL,
    compression_base=COMPRESSION_URL,
):
    pipeline_start_time_seconds = time.time()

    print("\n[Stage 1] OCR (%s)..." % ocr_base)
    with open(image_path, "rb") as image_file_handle:
        ocr_http_response = requests.post(
            "%s/ocr" % ocr_base,
            files={"image": image_file_handle},
            data={"cols": cols, "noise_profile": noise_profile},
            timeout=300,
        )
    ocr_http_response.raise_for_status()
    optical_character_recognition_result = ocr_http_response.json()
    extracted_text = optical_character_recognition_result["text"]
    print("  Extracted text   : %r" % extracted_text)
    print("  Noise profile    : %s" %
          optical_character_recognition_result["noise_profile"])
    print("  OCR latency      : %s ms" %
          optical_character_recognition_result["latency_ms"])

    if not extracted_text:
        raise ValueError("OCR returned empty text; cannot compress")

    print("\n[Stage 2] Adaptive Huffman (%s)..." % compression_base)
    compress_client_start_time_seconds = time.time()
    compress_http_response = requests.post(
        "%s/compress" % compression_base,
        json={"text": extracted_text},
        timeout=60,
    )
    compress_http_response.raise_for_status()
    compress_response_json = compress_http_response.json()
    compress_client_latency_milliseconds = (
        time.time() - compress_client_start_time_seconds
    ) * 1000
    compression_metrics = compress_response_json["metrics"]
    compressed_payload_hex = compress_response_json["data"]
    if len(compressed_payload_hex) > 40:
        compressed_hex_preview = compressed_payload_hex[:40] + "..."
    else:
        compressed_hex_preview = compressed_payload_hex
    print("  Compressed hex   : %s" % compressed_hex_preview)
    print("  Compress latency : %.2f ms (server: %s ms)" % (
        compress_client_latency_milliseconds,
        compress_response_json["latency_ms"],
    ))
    print("  Metrics:")
    print("    original bytes       : %s" %
          compression_metrics["original_bytes"])
    print("    compressed bytes     : %s" %
          compression_metrics["compressed_bytes"])
    print("    compression ratio    : %s" %
          compression_metrics["compression_ratio"])
    print("    entropy (bits/sym)   : %s" %
          compression_metrics["entropy_bits_per_symbol"])
    print("    avg bits/symbol      : %s" %
          compression_metrics["avg_bits_per_symbol"])
    print("    encoding efficiency  : %s" %
          compression_metrics["encoding_efficiency"])

    print("\n[Stage 2] Decompress...")
    decompress_client_start_time_seconds = time.time()
    decompress_http_response = requests.post(
        "%s/decompress" % compression_base,
        json={
            "data": compress_response_json["data"],
            "bit_length": compress_response_json["bit_length"],
        },
        timeout=60,
    )
    decompress_http_response.raise_for_status()
    decompress_response_json = decompress_http_response.json()
    recovered_text = decompress_response_json["text"]
    decompress_client_latency_milliseconds = (
        time.time() - decompress_client_start_time_seconds
    ) * 1000

    print("  Recovered text   : %r" % recovered_text)
    print("  Decompress latency: %.2f ms (server: %s ms)" % (
        decompress_client_latency_milliseconds,
        decompress_response_json["latency_ms"],
    ))

    lossless_round_trip_match = recovered_text == extracted_text
    if lossless_round_trip_match:
        print("\n  Lossless check   : PASS")
    else:
        print("\n  Lossless check   : FAIL")

    total_pipeline_latency_milliseconds = round(
        (time.time() - pipeline_start_time_seconds) * 1000, 2
    )
    print("\n[Pipeline] End-to-end latency: %s ms" %
          total_pipeline_latency_milliseconds)

    return {
        "text": extracted_text,
        "recovered_text": recovered_text,
        "lossless": lossless_round_trip_match,
        "metrics": compression_metrics,
        "ocr_latency_ms": optical_character_recognition_result["latency_ms"],
        "compress_client_ms": round(compress_client_latency_milliseconds, 2),
        "decompress_client_ms": round(decompress_client_latency_milliseconds, 2),
        "total_latency_ms": total_pipeline_latency_milliseconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument(
        "--noise-profile",
        default="none",
        choices=["none", "gaussian", "salt_and_pepper", "sidd"],
    )
    parser.add_argument("--ocr-url", default=OCR_URL)
    parser.add_argument("--compression-url", default=COMPRESSION_URL)
    args = parser.parse_args()

    result = run_pipeline(
        args.image,
        cols=args.cols,
        noise_profile=args.noise_profile,
        ocr_base=args.ocr_url.rstrip("/"),
        compression_base=args.compression_url.rstrip("/"),
    )
    print("\n" + json.dumps(result, indent=2))
