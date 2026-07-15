"""Byte-level vocabulary for RWKV.

Maps raw bytes 0x00-0xFF directly to token IDs, giving a fixed 256-token
vocab. 0x00 is reserved as PAD; 0x01 is UNK for control bytes we choose to
hide; everything else is a valid token.

Why this matters:
- No OOV, no tokenizer design choices, no vocabulary to tune.
- Directly compatible with BLT-style research (no patching yet, just no
  subword vocab).
- Character-level RWKV (current state) already ran at vocab=74; moving to
  bytes gets you the remaining 182 symbols for free.
"""

# ── Layout ──────────────────────────────────────────────────────────────────
PAD_ID = 0
UNK_ID = 1
FIRST_REAL_BYTE = 2

# Bytes 2..257 = 0x00..0xFF
BYTE_TO_ID: dict[int, int] = {b: FIRST_REAL_BYTE + b for b in range(256)}
ID_TO_BYTE: dict[int, int] = {v: k for k, v in BYTE_TO_ID.items()}
VOCAB_SIZE = FIRST_REAL_BYTE + 256  # 258 total


def encode(text: str, max_len: int = 512) -> list[int]:
    """Encode a string → list of ints in [0, vocab_size)."""
    tokens = [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [PAD_ID] * (max_len - len(tokens))
    return tokens


def decode(ids) -> str:
    """Decode list of ints back to string, skipping PAD/UNK."""
    chars = []
    for tid in ids:
        if tid in (PAD_ID, UNK_ID):
            continue
        b = ID_TO_BYTE.get(tid)
        if b is not None:
            chars.append(chr(b))
    return "".join(chars)


# Control bytes we treat as UNK (won't appear in normal training data)
_CONTROL_BYTES = frozenset(range(0, 32))  # 0x00-0x1F
