"""
Adaptive Huffman Encoding - Vitter's Algorithm
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    weight: int = 0
    symbol: Optional[str] = None      
    is_nyt: bool = False
    parent: Optional["Node"] = field(default=None, repr=False)
    left: Optional["Node"] = field(default=None, repr=False)
    right: Optional["Node"] = field(default=None, repr=False)
    order: int = 0                


class AdaptiveHuffmanTree:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.nyt = Node(weight=0, is_nyt=True, order=0)
        self.root = self.nyt
        self.symbol_nodes: dict[str, Node] = {}
        self._order_counter = 1
        self._blocks: dict[int, list[Node]] = {0: [self.nyt]}

    def _new_order(self) -> int:
        o = self._order_counter
        self._order_counter += 1
        return o

    def _block_add(self, node: Node):
        self._blocks.setdefault(node.weight, []).append(node)

    def _block_remove(self, node: Node):
        lst = self._blocks.get(node.weight, [])
        try:
            lst.remove(node)
        except ValueError:
            pass
        if not lst:
            self._blocks.pop(node.weight, None)


    def _path_to_root(self, node: Node) -> list[int]:
        bits = []
        while node.parent is not None:
            parent = node.parent
            bits.append(0 if parent.left is node else 1)
            node = parent
        return bits[::-1]

    def _find_highest_in_block(self, node: Node) -> Node:
        candidates = self._blocks.get(node.weight, [])
        best = node

        ancestors = set()
        anc = node.parent
        while anc is not None:
            ancestors.add(id(anc))
            anc = anc.parent

        for cur in candidates:
            if cur is node:
                continue
            if id(cur) in ancestors:
                continue
            if cur.order <= best.order:
                continue

            is_desc = False
            anc = cur.parent
            while anc is not None:
                if anc is node:
                    is_desc = True
                    break
                anc = anc.parent
            if not is_desc:
                best = cur
        return best

    def _swap_nodes(self, a: Node, b: Node):
        if a is b:
            return

        a_parent, b_parent = a.parent, b.parent

        if a_parent.left is a:
            a_parent.left = b
        else:
            a_parent.right = b

        if b_parent.left is b:
            b_parent.left = a
        else:
            b_parent.right = a

        a.parent, b.parent = b_parent, a_parent
        a.order, b.order = b.order, a.order

    def update(self, symbol: str):
        if symbol in self.symbol_nodes:
            node = self.symbol_nodes[symbol]
        else:
            internal = Node(
                weight=0,
                order=self.nyt.order,
                parent=self.nyt.parent,
            )
            new_symbol = Node(
                weight=0,
                symbol=symbol,
                order=self._new_order(),
                parent=internal,
            )
            new_nyt = Node(
                weight=0,
                is_nyt=True,
                order=self._new_order(),
                parent=internal,
            )
            internal.left = new_nyt
            internal.right = new_symbol

            if self.nyt.parent is not None:
                if self.nyt.parent.left is self.nyt:
                    self.nyt.parent.left = internal
                else:
                    self.nyt.parent.right = internal
            else:
                self.root = internal

            self._block_remove(self.nyt)
            self._block_add(internal)
            self._block_add(new_symbol)
            self._block_add(new_nyt)

            self.nyt = new_nyt
            self.symbol_nodes[symbol] = new_symbol
            node = new_symbol

        while node is not None:
            highest = self._find_highest_in_block(node)
            if (highest is not node
                    and highest is not node.parent
                    and node.parent is not None
                    and highest.parent is not None):
                self._swap_nodes(node, highest)

            self._block_remove(node)
            node.weight += 1
            self._block_add(node)
            node = node.parent


    def encode_symbol(self, symbol: str) -> list[int]:
        if symbol in self.symbol_nodes:
            return self._path_to_root(self.symbol_nodes[symbol])
        else:
            # NYT code + 8-bit ASCII of new symbol
            nyt_code = self._path_to_root(self.nyt)
            ascii_bits = [(ord(symbol) >> (7 - i)) & 1 for i in range(8)]
            return nyt_code + ascii_bits

    def decode_step(self, bits: list[int], pos: int) -> tuple[str, int]:

        node = self.root
        # If tree is empty (first symbol), read raw 8-bit ASCII
        if node is self.nyt:
            byte = 0
            for i in range(8):
                byte = (byte << 1) | bits[pos + i]
            return chr(byte), pos + 8

        while node.left is not None or node.right is not None:
            if pos >= len(bits):
                raise ValueError("Unexpected end of bitstream")
            bit = bits[pos]
            pos += 1
            node = node.left if bit == 0 else node.right

        if node.is_nyt:
            # Next 8 bits are the raw ASCII
            byte = 0
            for i in range(8):
                byte = (byte << 1) | bits[pos + i]
            return chr(byte), pos + 8

        return node.symbol, pos


def encode(text: str) -> tuple[bytes, int]:
    tree = AdaptiveHuffmanTree()
    bits = []
    for ch in text:
        bits.extend(tree.encode_symbol(ch))
        tree.update(ch)

    original_bit_length = len(bits)
    # Pad to byte boundary
    pad = (8 - len(bits) % 8) % 8
    bits.extend([0] * pad)

    compressed = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        compressed.append(byte)

    return bytes(compressed), original_bit_length


def decode(compressed: bytes, original_bit_length: int) -> str:
    bits = []
    for byte in compressed:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    bits = bits[:original_bit_length]

    tree = AdaptiveHuffmanTree()
    result = []
    pos = 0
    while pos < len(bits):
        symbol, pos = tree.decode_step(bits, pos)
        tree.update(symbol)
        result.append(symbol)

    return "".join(result)


def compute_metrics(text: str, compressed: bytes, original_bit_length: int) -> dict:
    original_bits = len(text) * 8
    compressed_bits = original_bit_length

    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 1.0

    # Shannon entropy
    from collections import Counter
    freq = Counter(text)
    n = len(text)
    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)

    # Encoding efficiency = entropy / avg_bits_per_symbol
    avg_bits = compressed_bits / n if n > 0 else 0
    efficiency = (entropy / avg_bits) if avg_bits > 0 else 0.0

    return {
        "original_bytes": len(text),
        "compressed_bytes": len(compressed),
        "compression_ratio": round(compression_ratio, 4),
        "entropy_bits_per_symbol": round(entropy, 4),
        "avg_bits_per_symbol": round(avg_bits, 4),
        "encoding_efficiency": round(efficiency, 4),
    }