"""Shared-State Architecture v2: True state sharing between byte_model and decoder.

The key insight: byte_model and decoder share the SAME RWKV state.
The decoder doesn't have its own RWKV blocks - it continues from byte_model's state.

Architecture:
1. byte_model: embed → RWKV blocks → byte_state (accumulated state)
2. patch_model: receives byte_state → outputs corrected_state + predicted_patch
3. decoder: takes corrected_state + patch info → predicts next byte (no separate RWKV)

This is true state sharing: the decoder operates on the same state trajectory as byte_model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rwkv_nano import RWKVBlock, count_params


class ByteLevelModel(nn.Module):
    """Processes bytes sequentially, accumulates state."""
    
    def __init__(self, vocab_size: int, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)  # Add prediction head
        
    def forward(self, tokens: torch.Tensor, state: dict | None = None):
        """
        Returns:
            byte_state: [B, T, dim] accumulated state at each position
            new_state: updated RWKV state (for sequential processing)
        """
        h = self.embed(tokens)
        
        for block in self.blocks:
            h, state = block(h, state)
        
        h = self.ln(h)
        return h, state


class PatchModel(nn.Module):
    """Receives byte_state, outputs corrected state + predicted patch (direction)."""
    
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        
        # Two output heads
        self.state_head = nn.Linear(dim, dim)  # Corrected byte_state
        self.direction_head = nn.Linear(dim, dim)  # Predicted next patch
        
    def forward(self, byte_state: torch.Tensor):
        """
        Args:
            byte_state: [B, dim] byte-level state (single position or pooled)
        Returns:
            corrected_state: [B, dim] corrected byte_state
            direction: [B, dim] predicted next patch (lookahead)
        """
        h = byte_state.unsqueeze(1)  # [B, 1, dim]
        
        for block in self.blocks:
            h, _ = block(h)
        
        h = self.ln(h)
        h = h.squeeze(1)  # [B, dim]
        
        corrected_state = self.state_head(h)
        direction = self.direction_head(h)
        
        return corrected_state, direction


class Decoder(nn.Module):
    """RWKV decoder that generates bytes sequentially.
    
    Accumulates state as it generates bytes. This state feeds back to encoder.
    """
    
    def __init__(self, vocab_size: int, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, corrected_state: torch.Tensor, direction: torch.Tensor, 
                tokens: torch.Tensor | None = None, state: dict | None = None):
        """
        Args:
            corrected_state: [B, dim] corrected state from patch_model
            direction: [B, dim] predicted next patch from patch_model
            tokens: [B, T] byte tokens for teacher forcing (optional)
            state: decoder's RWKV state (for sequential generation)
        Returns:
            logits: [B, vocab_size] or [B, T, vocab_size] next-byte prediction
            new_state: updated decoder state
        """
        # Combine corrected_state and direction as context
        context = corrected_state + direction  # [B, dim]
        
        if tokens is not None:
            # Training mode: process sequence with teacher forcing
            h = self.embed(tokens)  # [B, T, dim]
            # Add context to each position
            h = h + context.unsqueeze(1)  # [B, T, dim]
            
            # Process through RWKV blocks
            for block in self.blocks:
                h, state = block(h, state)
            
            h = self.ln(h)
            logits = self.head(h)  # [B, T, vocab_size]
            return logits, state
        else:
            # Generation mode: single step
            # Start with context as initial embedding
            h = context.unsqueeze(1)  # [B, 1, dim]
            
            # Process through RWKV blocks
            for block in self.blocks:
                h, state = block(h, state)
            
            h = self.ln(h)
            h = h.squeeze(1)  # [B, dim]
            logits = self.head(h)  # [B, vocab_size]
            return logits, state


class SharedStateModelV2(nn.Module):
    """Full recurrent architecture with three RWKV components.
    
    encoder-rwkv → patch-rwkv → decoder-rwkv
    decoder_state feeds back to encoder for next cycle.
    """
    
    def __init__(
        self,
        vocab_size: int = 258,
        dim: int = 64,
        n_byte_layers: int = 2,
        n_patch_layers: int = 1,
        n_decoder_layers: int = 2,
    ):
        super().__init__()
        self.dim = dim
        
        self.byte_model = ByteLevelModel(vocab_size, dim, n_byte_layers)
        self.patch_model = PatchModel(dim, n_patch_layers)
        self.decoder = Decoder(vocab_size, dim, n_decoder_layers)
        
    def forward(self, tokens: torch.Tensor, targets: torch.Tensor):
        """
        Training forward pass with full recurrent loop.
        
        1. Encoder processes bytes, accumulates state
        2. Patch-model transforms encoder state → corrected_state + direction
        3. Decoder uses corrected_state + direction, generates bytes, accumulates state
        4. Decoder state feeds back to encoder (via context)
        
        Args:
            tokens: [B, T] input byte tokens
            targets: [B, T] target byte tokens
        Returns:
            loss: combined loss (encoder + decoder)
            metrics: dict with individual losses
        """
        B, T = tokens.shape
        
        # Step 1: Encoder processes all bytes
        byte_state, _ = self.byte_model(tokens)  # [B, T, dim]
        
        # Step 2: Get final encoder state, send to patch-model
        final_byte_state = byte_state[:, -1, :]  # [B, dim]
        corrected_state, direction = self.patch_model(final_byte_state)  # [B, dim], [B, dim]
        
        # Step 3: Decoder generates bytes using corrected_state + direction
        # Decoder state will feed back to encoder in recurrent loop
        decoder_logits, _ = self.decoder(
            corrected_state, direction, tokens=tokens
        )  # [B, T, vocab_size]
        
        # Losses
        # Encoder loss (byte-level prediction)
        encoder_logits = self.byte_model.head(byte_state[:, :-1, :])  # [B, T-1, vocab]
        encoder_loss = F.cross_entropy(
            encoder_logits.reshape(-1, encoder_logits.size(-1)),
            targets[:, :-1].reshape(-1)
        )
        
        # Decoder loss (uses full context from patch-model)
        decoder_loss = F.cross_entropy(
            decoder_logits[:, :-1, :].reshape(-1, decoder_logits.size(-1)),
            targets[:, :-1].reshape(-1)
        )
        
        # Total loss
        total_loss = encoder_loss + decoder_loss
        
        metrics = {
            'encoder_loss': encoder_loss.item(),
            'decoder_loss': decoder_loss.item(),
        }
        
        return total_loss, metrics


if __name__ == "__main__":
    # Smoke test
    model = SharedStateModelV2(
        vocab_size=258,
        dim=64,
        n_byte_layers=2,
        n_patch_layers=1,
    )
    
    print(f"Total params: {count_params(model):,}")
    print(f"  byte_model: {count_params(model.byte_model):,}")
    print(f"  patch_model: {count_params(model.patch_model):,}")
    print(f"  decoder: {count_params(model.decoder):,}")
    
    B, T = 2, 64
    tokens = torch.randint(2, 258, (B, T))
    targets = torch.randint(2, 258, (B, T))
    
    loss, metrics = model(tokens, targets)
    print(f"\nForward pass:")
    print(f"  loss: {loss.item():.4f}")
    print(f"  metrics: {metrics}")
    
    # Gradient sanity
    loss.backward()
    n_zero_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0
    )
    n_total = sum(1 for _ in model.parameters())
    print(f"\nGradient sanity:")
    print(f"  {n_zero_grad}/{n_total} params have zero gradient")
    print("Done.")
