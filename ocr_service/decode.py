import torch
from config import CHARSET, BLANK

IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}


def encode_text(text):
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_ctc(logits):
    best = logits.argmax(dim=-1).cpu()
    out = []
    for b in range(best.shape[1]):
        chars, prev = [], BLANK
        for t in range(best.shape[0]):
            idx = int(best[t, b])
            if idx != BLANK and idx != prev and idx in IDX_TO_CHAR:
                chars.append(IDX_TO_CHAR[idx])
            prev = idx
        out.append("".join(chars))
    return out
