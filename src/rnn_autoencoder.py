"""RNN auto-encoder for byte sequences.

The encoder processes n bytes and compresses them into a fixed-size
hidden state vector. This hidden state IS the "n-gram embedding" — a
learned, continuous representation of the byte sequence.

Compared to BLT's hash n-grams:
- Hash: table lookup, discrete, collision-prone
- RNN auto-encoder: learned, continuous, no collisions, generalizes

The hidden state vector can be used as input to other models instead of
separate n-gram embeddings. It's a soft, learned compression of byte
sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.simple_rnn_receptance import SimpleRNNReceptance


class RNNAutoencoder(nn.Module):
    """RNN encoder-decoder for byte sequences.
    
    Args:
        n_gram: number of bytes per sequence (n-gram size)
        dim: hidden state dimension (also output embedding dim)
        vocab_size: byte vocabulary size
    """
    
    def __init__(self, n_gram: int, dim: int = 32, vocab_size: int = 258):
        super().__init__()
        self.n_gram = n_gram
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Encoder: bytes → hidden state
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder_rnn = SimpleRNNReceptance(dim, dim)
        
        # Decoder: hidden state → bytes
        self.decoder_embed = nn.Linear(dim, dim)
        self.decoder_rnn = SimpleRNNReceptance(dim, dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def encode(self, byte_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_seq: [batch, n_gram] byte token ids
        
        Returns:
            hidden_state: [batch, dim] compressed n-gram embedding
        """
        # Embed bytes
        x = self.embed(byte_seq)  # [batch, n_gram, dim]
        
        # Run encoder RNN, get final hidden state
        _, final_state, _ = self.encoder_rnn(x)
        return final_state  # [batch, dim]
    
    def decode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, dim] n-gram embedding
        
        Returns:
            reconstructed_bytes: [batch, n_gram, vocab_size] logits
        """
        batch_size = hidden_state.shape[0]
        
        # Use hidden state as input to decoder
        x = self.decoder_embed(hidden_state)  # [batch, dim]
        # Repeat for n_gram steps (decoder reads same context, predicts each byte)
        x = x.unsqueeze(1).expand(-1, self.n_gram, -1)  # [batch, n_gram, dim]
        
        # Run decoder RNN
        out, _, _ = self.decoder_rnn(x)
        
        # Project to vocab
        logits = self.head(out)  # [batch, n_gram, vocab_size]
        return logits
    
    def forward(self, byte_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Args:
            byte_seq: [batch, n_gram] byte token ids
        
        Returns:
            reconstructed_logits: [batch, n_gram, vocab_size]
            hidden_state: [batch, dim] the n-gram embedding
        """
        hidden_state = self.encode(byte_seq)
        reconstructed_logits = self.decode(hidden_state)
        return reconstructed_logits, hidden_state


def extract_ngram_windows(
    byte_stream: torch.Tensor,
    n_gram: int
) -> torch.Tensor:
    """Extract sliding n-gram windows from a byte stream.
    
    Args:
        byte_stream: [batch, seq_len] byte token ids
        n_gram: window size
    
    Returns:
        ngrams: [batch, n_windows, n_gram] byte token ids
    """
    batch_size, seq_len = byte_stream.shape
    n_windows = seq_len - n_gram + 1
    if n_windows <= 0:
        return torch.zeros(batch_size, 0, n_gram, device=byte_stream.device, dtype=byte_stream.dtype)
    
    ngrams = torch.zeros(batch_size, n_windows, n_gram, device=byte_stream.device, dtype=byte_stream.dtype)
    for i in range(n_windows):
        ngrams[:, i, :] = byte_stream[:, i:i + n_gram]
    return ngrams


if __name__ == "__main__":
    # Smoke test
    n_gram = 5
    dim = 32
    vocab_size = 258
    batch_size = 4
    
    model = RNNAutoencoder(n_gram=n_gram, dim=dim, vocab_size=vocab_size)
    
    # Generate fake byte sequences (n-gram sequences)
    byte_seq = torch.randint(1, 256, (batch_size, n_gram))
    
    # Forward pass
    reconstructed_logits, hidden_state = model(byte_seq)
    
    print(f"Input bytes: {byte_seq.shape}")
    print(f"Reconstructed logits: {reconstructed_logits.shape}")
    print(f"Hidden state (n-gram embedding): {hidden_state.shape}")
    
    # Test reconstruction loss
    target = byte_seq
    import torch.nn.functional as F
    loss = F.cross_entropy(
        reconstructed_logits.view(-1, vocab_size),
        target.view(-1),
        ignore_index=0,
    )
    print(f"Reconstruction loss: {loss.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    
    n_tensors = sum(1 for _ in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nParameters: {n_params} scalars across {n_tensors} tensors")
    print(f"Tensors with gradients: {n_grad}/{n_tensors}")
    print(f"Gradient flow: {'OK' if n_grad == n_tensors else 'ISSUE'}")
    
    # Test n-gram window extraction
    print(f"\n--- n-gram extraction test ---")
    long_stream = torch.randint(1, 256, (2, 10))
    ngrams = extract_ngram_windows(long_stream, n_gram=3)
    print(f"Input stream: {long_stream.shape}")
    print(f"Extracted 3-grams: {ngrams.shape}")
    print(f"First 3-gram of batch 0: {ngrams[0, 0].tolist()}")
    print(f"  (should be stream[0, 0:3]: {long_stream[0, 0:3].tolist()})")
    assert torch.equal(ngrams[0, 0], long_stream[0, 0:3])
    assert torch.equal(ngrams[0, 4], long_stream[0, 4:7])
    print("Extraction: OK")
