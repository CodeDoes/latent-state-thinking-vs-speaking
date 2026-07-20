"""Shared-State Architecture: byte-level + patch-model + decoder with lookahead.

Three coupled state vectors operating at different timescales:
1. Byte-level state (RWKV): processes bytes, accumulates state, tracks surprise
2. Patch-model: receives byte-level-state when surprised, outputs correction + lookahead
3. Decoder: uses byte-state + current patch + predicted patches to generate bytes

Key insight: byte-level-state IS the patch. When byte-level-model is surprised
(high entropy), it dumps its accumulated state to patch-model. No separate encoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.rwkv.rwkv_nano import RWKVBlock, count_params


class ByteLevelModel(nn.Module):
    """Processes bytes sequentially, accumulates state, detects surprise.
    
    Output: byte-state, surprise (entropy of next-byte prediction)
    """
    
    def __init__(self, vocab_size: int, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        # Head for next-byte prediction (to compute surprise)
        self.byte_head = nn.Linear(dim, vocab_size)
        
    def forward(self, tokens: torch.Tensor, state: dict | None = None):
        """
        Args:
            tokens: [B, T] byte tokens
            state: optional RWKV state from previous step
        Returns:
            byte_state: [B, T, dim] accumulated state
            surprise: [B, T] entropy of next-byte prediction
            new_state: updated RWKV state
        """
        h = self.embed(tokens)
        
        # Process through RWKV blocks
        for block in self.blocks:
            h, state = block(h, state)
        
        h = self.ln(h)
        
        # Compute next-byte logits for surprise detection
        logits = self.byte_head(h)  # [B, T, vocab_size]
        
        # Compute entropy (surprise) at each position
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        surprise = -(probs * log_probs).sum(dim=-1)  # [B, T]
        
        return h, surprise, state


class PatchModel(nn.Module):
    """Receives byte-level-state as patch, outputs correction + lookahead.
    
    Two output heads:
    1. State head: new byte-level-state (same dim as input)
    2. Direction head: predicted next patch (lookahead for decoder)
    """
    
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        
        # Two output heads
        self.state_head = nn.Linear(dim, dim)  # New byte-level-state
        self.direction_head = nn.Linear(dim, dim)  # Predicted next patch
        
    def forward(self, byte_state: torch.Tensor):
        """
        Args:
            byte_state: [B, dim] byte-level-state (the patch)
        Returns:
            new_state: [B, dim] corrected byte-level-state
            direction: [B, dim] predicted next patch (lookahead)
        """
        h = byte_state.unsqueeze(1)  # [B, 1, dim] - single patch
        
        # Process through RWKV blocks
        for block in self.blocks:
            h, _ = block(h)
        
        h = self.ln(h)
        h = h.squeeze(1)  # [B, dim]
        
        # Dual outputs
        new_state = self.state_head(h)
        direction = self.direction_head(h)
        
        return new_state, direction


class Decoder(nn.Module):
    """Generates bytes using byte-state + current patch + predicted patches.
    
    Input: byte-state + current patch + predicted next patch (concatenated)
    Output: next-byte logits + completion probability
    """
    
    def __init__(self, vocab_size: int, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        
        # Project concatenated input (3 * dim) to decoder dim
        self.input_proj = nn.Linear(3 * dim, dim)
        
        # Decoder blocks (RWKV for efficiency)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        
        # Output heads
        self.byte_head = nn.Linear(dim, vocab_size)
        self.completion_head = nn.Linear(dim, 1)  # Completion probability
        
    def forward(self, byte_state: torch.Tensor, current_patch: torch.Tensor, 
                predicted_patch: torch.Tensor, state: dict | None = None):
        """
        Args:
            byte_state: [B, T, dim] or [B, dim] byte-level-state
            current_patch: [B, T, dim] or [B, dim] current patch
            predicted_patch: [B, T, dim] or [B, dim] predicted next patch (lookahead)
            state: optional RWKV state
        Returns:
            logits: [B, T, vocab_size] or [B, vocab_size] next-byte prediction
            completion_prob: [B, T, 1] or [B, 1] probability of token completion
            new_state: updated RWKV state
        """
        # Handle both [B, dim] and [B, T, dim] inputs
        is_sequence = byte_state.dim() == 3
        
        # Concatenate all context
        h = torch.cat([byte_state, current_patch, predicted_patch], dim=-1)  # [B, T, 3*dim] or [B, 3*dim]
        h = self.input_proj(h)  # [B, T, dim] or [B, dim]
        
        # Add sequence dimension if needed for RWKV blocks
        if not is_sequence:
            h = h.unsqueeze(1)  # [B, 1, dim]
        
        # Process through decoder blocks
        for block in self.blocks:
            h, state = block(h, state)
        
        h = self.ln(h)
        
        # Remove sequence dimension if we added it
        if not is_sequence:
            h = h.squeeze(1)  # [B, dim]
        
        # Outputs
        logits = self.byte_head(h)  # [B, T, vocab_size] or [B, vocab_size]
        completion_prob = torch.sigmoid(self.completion_head(h))  # [B, T, 1] or [B, 1]
        
        return logits, completion_prob, state


class SharedStateModel(nn.Module):
    """Shared-state architecture with byte-level + patch-model + decoder.
    
    Training flow:
    1. Byte-level-model processes bytes, accumulates state, tracks surprise
    2. When surprise > threshold, send state to patch-model
    3. Patch-model outputs new state + predicted next patch
    4. Decoder generates bytes using all context
    """
    
    def __init__(
        self,
        vocab_size: int = 258,
        dim: int = 64,
        n_byte_layers: int = 2,
        n_patch_layers: int = 1,
        n_decoder_layers: int = 2,
        surprise_threshold: float = 3.0,  # Initial threshold (tunable)
    ):
        super().__init__()
        self.dim = dim
        self.surprise_threshold = surprise_threshold
        
        self.byte_model = ByteLevelModel(vocab_size, dim, n_byte_layers)
        self.patch_model = PatchModel(dim, n_patch_layers)
        self.decoder = Decoder(vocab_size, dim, n_decoder_layers)
        
    def forward(self, tokens: torch.Tensor, targets: torch.Tensor):
        """
        Training forward pass.
        
        Args:
            tokens: [B, T] input byte tokens
            targets: [B, T] target byte tokens (shifted by 1)
        Returns:
            loss: combined loss (byte + decoder)
            metrics: dict with individual losses
        """
        B, T = tokens.shape
        
        # Step 1: Byte-level-model processes all bytes
        byte_state, surprise, _ = self.byte_model(tokens)  # [B, T, dim], [B, T]
        
        # Pool byte_state to create patch representation (use mean for now)
        current_patch = byte_state.mean(dim=1)  # [B, dim]
        
        # Step 2: Patch-model processes the patch
        new_state, predicted_patch = self.patch_model(current_patch)  # [B, dim], [B, dim]
        
        # Step 3: Decoder predicts next byte at each position
        # Expand patch info to all positions
        current_patch_expanded = current_patch.unsqueeze(1).expand(-1, T, -1)  # [B, T, dim]
        predicted_patch_expanded = predicted_patch.unsqueeze(1).expand(-1, T, -1)  # [B, T, dim]
        
        # Decoder takes byte_state + current_patch + predicted_patch
        decoder_logits, completion_prob, _ = self.decoder(
            byte_state, current_patch_expanded, predicted_patch_expanded
        )  # [B, T, vocab_size], [B, T, 1]
        
        # Losses
        # Byte-level loss (next-byte prediction from byte_model)
        byte_logits = self.byte_model.byte_head(byte_state)  # [B, T, vocab]
        byte_loss = F.cross_entropy(
            byte_logits.reshape(-1, byte_logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Decoder loss (next-byte prediction using all context)
        decoder_loss = F.cross_entropy(
            decoder_logits.reshape(-1, decoder_logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Total loss: byte_model + decoder
        total_loss = byte_loss + decoder_loss
        
        avg_surprise = surprise.mean(dim=1)  # [B]
        
        metrics = {
            'byte_loss': byte_loss.item(),
            'decoder_loss': decoder_loss.item(),
            'avg_surprise': avg_surprise.mean().item(),
        }
        
        return total_loss, metrics


if __name__ == "__main__":
    # Smoke test
    model = SharedStateModel(
        vocab_size=258,
        dim=64,
        n_byte_layers=2,
        n_patch_layers=1,
        n_decoder_layers=2,
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
