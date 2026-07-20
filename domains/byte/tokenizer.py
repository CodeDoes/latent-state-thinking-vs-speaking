"""Minimal symbol tokenizer for the fast synthetic reasoning task."""


class SymbolTokenizer:
    def __init__(self, vocab):
        self.special = ["<PAD>", "<UNK>"]
        self.vocab = self.special + list(vocab)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)
        self.pad = self.stoi["<PAD>"]

    def encode(self, toks):
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in toks]

    def decode(self, ids):
        return [self.itos.get(int(i), "<UNK>") for i in ids]
