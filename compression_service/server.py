"""
Compression Microservice - Stage 2
POST /compress   : accepts JSON {"text": "..."}, returns compressed bytes + metrics
POST /decompress : accepts JSON {"data": "<hex>", "bit_length": N}, returns original text
GET  /health     : liveness check
"""

import time
from flask import Flask, request, jsonify
from huffman import encode, decode, compute_metrics

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/compress", methods=["POST"])
def compress():
    body = request.get_json(force=True)
    if "text" not in body:
        return jsonify({"error": "missing 'text' field"}), 400

    text = body["text"]
    if not text:
        return jsonify({"error": "empty text"}), 400

    t0 = time.time()
    compressed, bit_length = encode(text)
    latency_ms = (time.time() - t0) * 1000

    metrics = compute_metrics(text, compressed, bit_length)

    return jsonify({
        "data": compressed.hex(),
        "bit_length": bit_length,
        "metrics": metrics,
        "latency_ms": round(latency_ms, 2),
    })


@app.route("/decompress", methods=["POST"])
def decompress():
    body = request.get_json(force=True)
    if "data" not in body or "bit_length" not in body:
        return jsonify({"error": "missing 'data' or 'bit_length'"}), 400

    t0 = time.time()
    compressed = bytes.fromhex(body["data"])
    text = decode(compressed, body["bit_length"])
    latency_ms = (time.time() - t0) * 1000

    return jsonify({
        "text": text,
        "latency_ms": round(latency_ms, 2),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)