# Adaptive Huffman coding from scratch.
#
# Idea: encoder and decoder both maintain the exact same symbol-frequency
# counter. Before each symbol, both rebuild a Huffman code table from the
# current counts. The encoder emits that symbol's code; the decoder walks the
# tree to recover it. Then both increment the count for that symbol and move
# on. A dedicated NYT ("not yet transmitted") code handles first appearances:
# when the encoder hits an unseen character it emits the NYT code followed by
# the raw 8-bit byte, and the decoder does the inverse.
#
# This is genuinely online/adaptive — no header, no second pass.

import heapq
import math
from collections import Counter

NYT = -1  # sentinel for "not yet transmitted"


class _Node:
    __slots__ = ("freq", "symbol", "left", "right", "order")

    def __init__(self, freq, symbol=None, left=None, right=None, order=0):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.order = order  # tie-breaker so heap comparisons are deterministic

    def __lt__(self, other):
        return (self.freq, self.order) < (other.freq, other.order)


def _build_tree(freqs):
    """Build a Huffman tree from {symbol: count}. Symbols = ints (char codes)
    or the NYT sentinel. Always includes NYT so new symbols stay encodable."""
    items = list(freqs.items())
    if NYT not in freqs:
        items.append((NYT, 0))

    heap = [_Node(f, s, order=i) for i, (s, f) in enumerate(items)]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(a.freq + b.freq, None, a, b, counter))
        counter += 1
    return heap[0]


def _codes_from_tree(root):
    """Walk the tree, collect {symbol: list_of_bits}. Root with a single leaf
    gets the code [0]."""
    codes = {}

    def walk(node, path):
        if node.left is None and node.right is None:
            codes[node.symbol] = path or [0]
            return
        walk(node.left, path + [0])
        walk(node.right, path + [1])

    walk(root, [])
    return codes


def _byte_bits(n):
    return [(n >> (7 - i)) & 1 for i in range(8)]


def encode(text):
    """Returns (packed_bytes, total_bit_length_without_padding)."""
    freqs = {}
    bits = []
    for ch in text:
        code_point = ord(ch)
        tree = _build_tree(freqs)
        codes = _codes_from_tree(tree)
        if code_point in freqs:
            bits.extend(codes[code_point])
        else:
            bits.extend(codes[NYT])
            bits.extend(_byte_bits(code_point))
        freqs[code_point] = freqs.get(code_point, 0) + 1

    bit_len = len(bits)
    # pad up to a byte boundary
    bits.extend([0] * ((8 - bit_len % 8) % 8))
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out), bit_len


def decode(blob, bit_len):
    """Inverse of encode: needs the exact unpadded bit length."""
    # unpack bytes -> bit list, trimmed back to real length
    bits = []
    for byte in blob:
        bits.extend((byte >> (7 - i)) & 1 for i in range(8))
    bits = bits[:bit_len]

    freqs = {}
    out = []
    i = 0
    while i < len(bits):
        tree = _build_tree(freqs)
        node = tree
        # descend to a leaf (handles the single-leaf edge case)
        if node.left is None and node.right is None:
            symbol = node.symbol
            i += 1  # the one bit we emitted for a single-leaf tree
        else:
            while node.left is not None:
                node = node.left if bits[i] == 0 else node.right
                i += 1
            symbol = node.symbol

        if symbol == NYT:
            code_point = 0
            for _ in range(8):
                code_point = (code_point << 1) | bits[i]
                i += 1
            out.append(chr(code_point))
            freqs[code_point] = freqs.get(code_point, 0) + 1
        else:
            out.append(chr(symbol))
            freqs[symbol] = freqs.get(symbol, 0) + 1
    return "".join(out)


def compute_metrics(text, blob, bit_len):
    """Basic stats the server returns alongside the compressed bytes."""
    original_bits = len(text) * 8
    ratio = original_bits / bit_len if bit_len else 1.0

    counts = Counter(text)
    n = len(text)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) if n else 0.0
    avg_bits = bit_len / n if n else 0.0
    efficiency = entropy / avg_bits if avg_bits else 0.0

    return {
        "original_bytes": len(text),
        "compressed_bytes": len(blob),
        "compression_ratio": round(ratio, 4),
        "entropy_bits_per_symbol": round(entropy, 4),
        "avg_bits_per_symbol": round(avg_bits, 4),
        "encoding_efficiency": round(efficiency, 4),
    }
