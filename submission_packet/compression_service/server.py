# Stage 2: POST /compress, POST /decompress, GET /health — adaptive Huffman in huffman.py

import os
import time
from flask import Flask, request, jsonify
from huffman import encode, decode, compute_metrics

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/compress", methods=["POST"])
def compress():
    request_json_body = request.get_json(force=True)
    if "text" not in request_json_body:
        return jsonify({"error": "missing 'text' field"}), 400

    plain_text = request_json_body["text"]
    if not plain_text:
        return jsonify({"error": "empty text"}), 400

    request_start_time_seconds = time.time()
    compressed_bytes, encoded_bit_length = encode(plain_text)
    server_latency_milliseconds = (time.time() - request_start_time_seconds) * 1000

    metrics = compute_metrics(plain_text, compressed_bytes, encoded_bit_length)

    return jsonify({
        "data": compressed_bytes.hex(),
        "bit_length": encoded_bit_length,
        "metrics": metrics,
        "latency_ms": round(server_latency_milliseconds, 2),
    })


@app.route("/decompress", methods=["POST"])
def decompress():
    request_json_body = request.get_json(force=True)
    if "data" not in request_json_body or "bit_length" not in request_json_body:
        return jsonify({"error": "missing 'data' or 'bit_length'"}), 400

    request_start_time_seconds = time.time()
    compressed_bytes = bytes.fromhex(request_json_body["data"])
    recovered_plain_text = decode(
        compressed_bytes, request_json_body["bit_length"]
    )
    server_latency_milliseconds = (time.time() - request_start_time_seconds) * 1000

    return jsonify({
        "text": recovered_plain_text,
        "latency_ms": round(server_latency_milliseconds, 2),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
