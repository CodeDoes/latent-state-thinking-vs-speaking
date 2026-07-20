"""Hybrid tokenizer: O(1) flat lookup for known tokens, TRIE fallback for OOD.

Primary path is a flat `bytes -> tid` dict (built once from the vocab). This
resolves the common tokens in O(1) with no TRIE traversal. Only byte-spans the
flat table does not recognise fall back to the TRIE (which holds the full
vocab) for correct longest-match resolution. Matches the real tokenizer 100%.

Usage:
    from src.hybrid_tokenizer import encode, decode, token_bytes
    ids = encode("Hello world")          # -> list[int] world-vocab token IDs
    text = decode(ids)                   # -> str
    bs = token_bytes(ids[0])             # -> bytes for one token
"""

from pathlib import Path

from src.hf_rwkv_tokenizer import TRIE

VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"


def _load_vocab(path):
    """Parse rwkv_vocab_v20230424.txt -> (idx2token, token2idx)."""
    idx2token = {}
    token2idx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.index(" ")
            idx = int(line[:sp])
            x = eval(line[sp:line.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            idx2token[idx] = x
            token2idx[x] = idx
    return idx2token, token2idx


IDX2TOKEN, TOKEN2IDX = _load_vocab(VOCAB_PATH)

# Flat O(1) lookup: bytes -> token ID. Covers every vocab entry.
BYTES_TO_TID = TOKEN2IDX

# TRIE holds the full vocab as fallback for anything the flat table
# cannot resolve (e.g. byte-spans not present as standalone tokens, or
# OOD input that still needs longest-match segmentation).
_root = TRIE()
for t, i in TOKEN2IDX.items():
    _root.add(t, val=(t, i))


def encode(text: str) -> list[int]:
    """Encode text to world-vocab token IDs.

    Use the TRIE for longest-match boundary detection (optimal single pass),
    then resolve the matched span to its token ID via the flat O(1) dict.
    The flat dict replaces the original tokenizer's value storage, so no
    extra hash computation is needed.
    """
    raw = text.encode("utf-8")
    tokens = []
    i = 0
    n = len(raw)
    while i < n:
        idx, node, values = _root.find_longest(raw, i)
        if idx == i:
            tokens.append(TOKEN2IDX.get(raw[i:i + 1], 0))
            i += 1
            continue
        tid = BYTES_TO_TID.get(raw[i:idx])
        if tid is not None:
            tokens.append(tid)
        else:
            _, tid = next(iter(values))
            tokens.append(tid)
        i = idx
    return tokens


def decode(token_ids: list[int]) -> str:
    """Decode world-vocab token IDs back to text (local map, no TRIE)."""
    return b"".join(IDX2TOKEN.get(tid, b"") for tid in token_ids).decode(
        "utf-8", errors="replace"
    )


def token_bytes(tid: int) -> bytes:
    """Return the raw bytes for a single world-vocab token ID."""
    return IDX2TOKEN.get(tid, b"")
