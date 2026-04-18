# Adaptive Huffman (FGK / Vitter): tree updates per symbol; NYT = not-yet-transmitted escape.

import math
from collections import Counter


class HuffmanTreeNode:
    """One node in the adaptive Huffman tree (internal split or leaf)."""

    def __init__(
        self,
        symbol_frequency=0,
        symbol_character=None,
        is_not_yet_transmitted_escape=False,
        tree_order_index=0,
        parent_node=None,
    ):
        self.symbol_frequency = symbol_frequency
        self.symbol_character = symbol_character
        self.is_not_yet_transmitted_escape = is_not_yet_transmitted_escape
        self.parent_node = parent_node
        self.left_child = None
        self.right_child = None
        self.tree_order_index = tree_order_index


class AdaptiveHuffmanTree:
    """Vitter-style adaptive coding: new symbols emit NYT path plus 8-bit ASCII."""

    def __init__(self):
        self.not_yet_transmitted_escape_node = HuffmanTreeNode(
            symbol_frequency=0,
            is_not_yet_transmitted_escape=True,
            tree_order_index=0,
        )
        self.root_node = self.not_yet_transmitted_escape_node
        self.symbol_character_to_leaf_node = {}
        self._next_tree_order_index = 1

    def _allocate_next_tree_order_index(self):
        assigned_index = self._next_tree_order_index
        self._next_tree_order_index += 1
        return assigned_index

    def _collect_bits_from_root_to_node(self, target_node):
        """Bits from root to leaf: 0 = left_child, 1 = right_child."""
        bits_from_root_to_leaf = []
        current_node = target_node
        while current_node.parent_node is not None:
            parent_node = current_node.parent_node
            if parent_node.left_child is current_node:
                bits_from_root_to_leaf.append(0)
            else:
                bits_from_root_to_leaf.append(1)
            current_node = parent_node
        bits_from_root_to_leaf.reverse()
        return bits_from_root_to_leaf

    def _collect_all_nodes_preorder(self):
        all_nodes = []
        stack = [self.root_node]
        while len(stack) > 0:
            current_node = stack.pop()
            all_nodes.append(current_node)
            if current_node.left_child is not None:
                stack.append(current_node.left_child)
            if current_node.right_child is not None:
                stack.append(current_node.right_child)
        return all_nodes

    def _is_first_node_ancestor_of_second(self, potential_ancestor, descendant):
        walk_node = descendant.parent_node
        while walk_node is not None:
            if walk_node is potential_ancestor:
                return True
            walk_node = walk_node.parent_node
        return False

    def _find_highest_tree_order_in_same_frequency_block(self, node):
        block_frequency = node.symbol_frequency
        best_candidate_node = node
        for candidate_node in self._collect_all_nodes_preorder():
            if candidate_node.symbol_frequency != block_frequency or candidate_node is node:
                continue
            if self._is_first_node_ancestor_of_second(candidate_node, node):
                continue
            if self._is_first_node_ancestor_of_second(node, candidate_node):
                continue
            if candidate_node.tree_order_index > best_candidate_node.tree_order_index:
                best_candidate_node = candidate_node
        return best_candidate_node

    def _swap_two_subtree_nodes(self, first_node, second_node):
        if first_node is second_node:
            return
        first_parent = first_node.parent_node
        second_parent = second_node.parent_node
        if first_parent is not None and first_parent is second_parent:
            shared_parent = first_parent
            if shared_parent.left_child is first_node and shared_parent.right_child is second_node:
                shared_parent.left_child, shared_parent.right_child = (
                    second_node,
                    first_node,
                )
            elif shared_parent.left_child is second_node and shared_parent.right_child is first_node:
                shared_parent.left_child, shared_parent.right_child = (
                    first_node,
                    second_node,
                )
            first_node.parent_node = shared_parent
            second_node.parent_node = shared_parent
            first_node.tree_order_index, second_node.tree_order_index = (
                second_node.tree_order_index,
                first_node.tree_order_index,
            )
            return
        if first_parent.left_child is first_node:
            first_parent.left_child = second_node
        else:
            first_parent.right_child = second_node
        if second_parent.left_child is second_node:
            second_parent.left_child = first_node
        else:
            second_parent.right_child = first_node
        first_node.parent_node, second_node.parent_node = second_parent, first_parent
        first_node.tree_order_index, second_node.tree_order_index = (
            second_node.tree_order_index,
            first_node.tree_order_index,
        )

    def update_after_symbol(self, symbol_character):
        if symbol_character in self.symbol_character_to_leaf_node:
            current_node = self.symbol_character_to_leaf_node[symbol_character]
        else:
            internal_split_node = HuffmanTreeNode(
                symbol_frequency=0,
                tree_order_index=self.not_yet_transmitted_escape_node.tree_order_index,
                parent_node=self.not_yet_transmitted_escape_node.parent_node,
            )
            new_symbol_leaf_node = HuffmanTreeNode(
                symbol_frequency=0,
                symbol_character=symbol_character,
                tree_order_index=self._allocate_next_tree_order_index(),
                parent_node=internal_split_node,
            )
            new_escape_leaf_node = HuffmanTreeNode(
                symbol_frequency=0,
                is_not_yet_transmitted_escape=True,
                tree_order_index=self._allocate_next_tree_order_index(),
                parent_node=internal_split_node,
            )
            internal_split_node.left_child = new_escape_leaf_node
            internal_split_node.right_child = new_symbol_leaf_node
            if self.not_yet_transmitted_escape_node.parent_node is not None:
                parent_of_escape = self.not_yet_transmitted_escape_node.parent_node
                if parent_of_escape.left_child is self.not_yet_transmitted_escape_node:
                    parent_of_escape.left_child = internal_split_node
                else:
                    parent_of_escape.right_child = internal_split_node
            else:
                self.root_node = internal_split_node
            self.not_yet_transmitted_escape_node = new_escape_leaf_node
            self.symbol_character_to_leaf_node[symbol_character] = new_symbol_leaf_node
            current_node = new_symbol_leaf_node

        while current_node is not None:
            highest_in_block = self._find_highest_tree_order_in_same_frequency_block(
                current_node
            )
            if highest_in_block is not current_node and highest_in_block is not current_node.parent_node:
                if current_node.parent_node is not None and highest_in_block.parent_node is not None:
                    self._swap_two_subtree_nodes(current_node, highest_in_block)
            current_node.symbol_frequency += 1
            current_node = current_node.parent_node

    def encode_symbol_to_bits(self, symbol_character):
        if symbol_character in self.symbol_character_to_leaf_node:
            return self._collect_bits_from_root_to_node(
                self.symbol_character_to_leaf_node[symbol_character]
            )
        not_yet_transmitted_bits = self._collect_bits_from_root_to_node(
            self.not_yet_transmitted_escape_node
        )
        ascii_bits_eight = []
        character_code_point = ord(symbol_character)
        for bit_position in range(8):
            ascii_bits_eight.append((character_code_point >> (7 - bit_position)) & 1)
        return not_yet_transmitted_bits + ascii_bits_eight

    def decode_one_symbol_from_bitstream(self, bitstream_bits, start_bit_index):
        current_node = self.root_node
        if current_node is self.not_yet_transmitted_escape_node:
            decoded_byte = 0
            for offset in range(8):
                decoded_byte = (decoded_byte << 1) | bitstream_bits[start_bit_index + offset]
            return chr(decoded_byte), start_bit_index + 8
        while current_node.left_child is not None or current_node.right_child is not None:
            if start_bit_index >= len(bitstream_bits):
                raise ValueError("truncated bitstream")
            next_bit = bitstream_bits[start_bit_index]
            start_bit_index += 1
            if next_bit == 0:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        if current_node.is_not_yet_transmitted_escape:
            decoded_byte = 0
            for offset in range(8):
                decoded_byte = (decoded_byte << 1) | bitstream_bits[start_bit_index + offset]
            return chr(decoded_byte), start_bit_index + 8
        return current_node.symbol_character, start_bit_index


def encode(plain_text):
    adaptive_tree = AdaptiveHuffmanTree()
    encoded_bits = []
    for character in plain_text:
        encoded_bits.extend(adaptive_tree.encode_symbol_to_bits(character))
        adaptive_tree.update_after_symbol(character)
    total_encoded_bit_count = len(encoded_bits)
    padding_bit_count = (8 - total_encoded_bit_count % 8) % 8
    for _ in range(padding_bit_count):
        encoded_bits.append(0)
    packed_bytes = bytearray()
    bit_index = 0
    while bit_index < len(encoded_bits):
        packed_byte = 0
        for inner in range(8):
            packed_byte = (packed_byte << 1) | encoded_bits[bit_index + inner]
        packed_bytes.append(packed_byte)
        bit_index += 8
    return bytes(packed_bytes), total_encoded_bit_count


def decode(compressed_bytes, original_bit_length_without_padding):
    bitstream_bits = []
    for single_byte in compressed_bytes:
        for bit_position in range(7, -1, -1):
            bitstream_bits.append((single_byte >> bit_position) & 1)
    bitstream_bits = bitstream_bits[:original_bit_length_without_padding]

    adaptive_tree = AdaptiveHuffmanTree()
    decoded_characters = []
    current_bit_index = 0
    while current_bit_index < len(bitstream_bits):
        decoded_symbol, current_bit_index = adaptive_tree.decode_one_symbol_from_bitstream(
            bitstream_bits, current_bit_index
        )
        adaptive_tree.update_after_symbol(decoded_symbol)
        decoded_characters.append(decoded_symbol)
    return "".join(decoded_characters)


def compute_metrics(plain_text, compressed_bytes, original_bit_length_without_padding):
    original_bit_count = len(plain_text) * 8
    compressed_bit_count = original_bit_length_without_padding
    if compressed_bit_count > 0:
        compression_ratio = original_bit_count / compressed_bit_count
    else:
        compression_ratio = 1.0

    symbol_frequency_counter = Counter(plain_text)
    text_length = len(plain_text)
    shannon_entropy_bits_per_symbol = 0.0
    for character in symbol_frequency_counter:
        probability = symbol_frequency_counter[character] / text_length
        shannon_entropy_bits_per_symbol -= probability * math.log2(probability)

    average_encoded_bits_per_symbol = (
        compressed_bit_count / text_length if text_length > 0 else 0.0
    )
    if average_encoded_bits_per_symbol > 0:
        encoding_efficiency = shannon_entropy_bits_per_symbol / average_encoded_bits_per_symbol
    else:
        encoding_efficiency = 0.0

    return {
        "original_bytes": len(plain_text),
        "compressed_bytes": len(compressed_bytes),
        "compression_ratio": round(compression_ratio, 4),
        "entropy_bits_per_symbol": round(shannon_entropy_bits_per_symbol, 4),
        "avg_bits_per_symbol": round(average_encoded_bits_per_symbol, 4),
        "encoding_efficiency": round(encoding_efficiency, 4),
    }
