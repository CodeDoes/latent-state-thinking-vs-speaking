"""
Simple character-level tokenizer for synthetic tasks.

Since we're generating our own text, we don't need a heavy tokenizer.
Character-level is sufficient for the toy world tasks and keeps vocab small.
"""

from collections import Counter
from typing import List, Dict, Optional, Tuple
import json
import os


class CharTokenizer:
    """Character-level tokenizer with special tokens."""

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        max_vocab: int = 256,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        eos_token: str = "<EOS>",
        sep_token: str = "<SEP>",
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.sep_token = sep_token

        # Build vocabulary
        self.char_counts = Counter()
        if texts:
            for text in texts:
                self.char_counts.update(text)

        # Special tokens first
        self.special_tokens = [pad_token, unk_token, eos_token, sep_token]
        self.vocab = {c: idx + len(self.special_tokens) for idx, (c, _) in enumerate(self.char_counts.most_common(max_vocab - len(self.special_tokens)))}
        for i, tok in enumerate(self.special_tokens):
            self.vocab[tok] = i

        self.inv_vocab = {i: c for c, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        ids = [self.vocab.get(c, self.vocab[self.unk_token]) for c in text]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.inv_vocab.get(i, self.unk_token) for i in ids)

    def save(self, path: str):
        """Save vocabulary to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "vocab": self.vocab,
                "special_tokens": self.special_tokens,
                "vocab_size": self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.special_tokens = data["special_tokens"]
        tok.inv_vocab = {i: c for c, i in tok.vocab.items()}
        tok.vocab_size = len(tok.vocab)
        return tok


def build_tokenizer_from_dataset(dataset: List[dict], max_vocab: int = 256) -> CharTokenizer:
    """Build tokenizer from dataset samples.

    Includes the QA formatting markers (Question:, Answer:, newline, ':') so
    they are never mapped to <UNK> when the strict "Answer: <X>" surface form
    is used during training/evaluation.
    """
    texts = []
    for sample in dataset:
        texts.append(sample["narrative"])
        if sample.get("question"):
            texts.append(sample["question"])
        texts.append(sample["answer"])
    # Ensure format markers are present in the vocabulary.
    texts += ["Question:", "Answer:", "\n", ":", "The secret code for was updated to"]
    return CharTokenizer(texts, max_vocab=max_vocab)
