"""RNN Patch Model — hierarchical byte/patch/byte autoencoder.

Architecture (theories/encoder-decoder-patch.md):

Phase 1 (autoencoder):
    encoder: sequence of N bytes → patch[D]
    decoder: patch[D] → sequence of N bytes
    Loss: next-byte prediction (cross-entropy)
    Train encoder+decoder as one autoencoder

Phase 2 (hierarchical):
    Freeze encoder+decoder
    patch-model: patch[D] → transformed_patch[D]
    Loss: next-byte prediction through the transformed patch
    Train patch-model only

States:
    byte-level: encoder and decoder (implicit in their recurrence)
    patch: D floats (= encoder output = decoder input)
    patch-level-state: Dp floats (patch-model's recurrent state)

Prove one thing: hierarchical prediction enables longer-range prediction
than flat byte-level prediction at matched parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


BYTE_STATE_DIM = 8  # encoder/decoder hidden dimension
PATCH_DIM = 8       # patch dimension
VOCAB_SIZE = 256    # byte vocabulary


class ByteGRUCell(nn.Module):
    """Simple GRU cell for byte-level processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU gates
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: [B, input_dim]
            hidden: [B, hidden_dim]
        Returns:
            new_hidden: [B, hidden_dim]
        """
        combined = torch.cat([input, hidden], dim=-1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        
        candidate_input = torch.cat([input, reset * hidden], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input))
        
        new_hidden = (1 - update) * hidden + update * candidate
        return new_hidden


class EncoderModel(nn.Module):
    """Byte-level encoder: compresses a sequence of bytes into a patch vector.
    
    Input: sequence of N bytes [B, N]
    Output: patch vector [B, D]
    """
    
    def __init__(self, dim: int = BYTE_STATE_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.cell = ByteGRUCell(input_dim=dim, hidden_dim=dim)
        
    def forward(self, bytes_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bytes_seq: [B, N] byte indices
        Returns:
            patch: [B, D] compressed representation
        """
        B, N = bytes_seq.shape
        hidden = torch.zeros(B, self.dim, device=bytes_seq.device)
        
        for i in range(N):
            byte_embed = self.embed(bytes_seq[:, i])  # [B, D]
            hidden = self.cell(byte_embed, hidden)
        
        return hidden  # This is the patch


class DecoderModel(nn.Module):
    """Byte-level decoder: expands a patch vector into a sequence of bytes.
    
    Input: patch vector [B, D]
    Output: byte logits [B, N, V]
    """
    
    def __init__(self, dim: int = BYTE_STATE_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Decoder uses patch as input at each step (autoregressive)
        self.cell = ByteGRUCell(input_dim=dim, hidden_dim=dim)
        self.byte_head = nn.Linear(dim, vocab_size)
        
    def forward(self, patch: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Args:
            patch: [B, D] compressed representation
            target_length: N, number of bytes to generate
        Returns:
            logits: [B, N, V] byte predictions
        """
        B = patch.shape[0]
        hidden = torch.zeros(B, self.dim, device=patch.device)
        
        logits_list = []
        for _ in range(target_length):
            # Use patch as input at each step (like conditioning)
            hidden = self.cell(patch, hidden)
            logits = self.byte_head(hidden)  # [B, V]
            logits_list.append(logits)
        
        logits = torch.stack(logits_list, dim=1)  # [B, N, V]
        return logits


class PatchModel(nn.Module):
    """Patch-level model: transforms patch vectors.
    
    Input: patch[D]
    Output: transformed_patch[D]
    
    Maintains a recurrent state across patches for context.
    """
    
    def __init__(self, patch_dim: int = PATCH_DIM, patch_state_dim: int = 16):
        super().__init__()
        self.patch_dim = patch_dim
        self.patch_state_dim = patch_state_dim
        
        # GRU cell for patch-level recurrence
        self.cell = ByteGRUCell(input_dim=patch_dim, hidden_dim=patch_state_dim)
        
        # Transform patch based on recurrent state
        self.transform = nn.Linear(patch_dim + patch_state_dim, patch_dim)
        
    def forward(
        self,
        patch: torch.Tensor,
        patch_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch: [B, D] input patch
            patch_state: [B, Dp] recurrent state
        Returns:
            transformed_patch: [B, D] transformed patch
            new_patch_state: [B, Dp] updated recurrent state
        """
        new_patch_state = self.cell(patch, patch_state)
        combined = torch.cat([patch, new_patch_state], dim=-1)
        transformed_patch = self.transform(combined)
        
        return transformed_patch, new_patch_state


class RNXPatchModel(nn.Module):
    """Full hierarchical model: encoder + decoder + patch-model.
    
    Phase 1: Train encoder+decoder as autoencoder
    Phase 2: Freeze encoder+decoder, train patch-model
    """
    
    def __init__(
        self,
        byte_state_dim: int = BYTE_STATE_DIM,
        patch_dim: int = PATCH_DIM,
        patch_state_dim: int = 16,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.byte_state_dim = byte_state_dim
        self.patch_dim = patch_dim
        self.patch_state_dim = patch_state_dim
        self.vocab_size = vocab_size
        
        self.encoder = EncoderModel(byte_state_dim, vocab_size)
        self.decoder = DecoderModel(byte_state_dim, vocab_size)
        self.patch_model = PatchModel(patch_dim, patch_state_dim)
        
    def forward_phase1(
        self,
        bytes_seq: torch.Tensor,
        patch_size: int = 8,
    ) -> torch.Tensor:
        """Phase 1: autoencoder (encoder + decoder).
        
        Split sequence into patches of size patch_size.
        For each patch:
            - Encode patch_size bytes → patch vector
            - Decode patch vector → patch_size bytes
        Concatenate all reconstructed patches.
        
        Args:
            bytes_seq: [B, T] input bytes
            patch_size: N, bytes per patch
        Returns:
            logits: [B, T, V] reconstructed byte logits
        """
        B, T = bytes_seq.shape
        device = bytes_seq.device
        
        all_logits = []
        
        # Process in fixed-size patches
        for patch_start in range(0, T, patch_size):
            patch_end = min(patch_start + patch_size, T)
            actual_patch_size = patch_end - patch_start
            
            # Encode this patch
            patch_bytes = bytes_seq[:, patch_start:patch_end]
            patch = self.encoder(patch_bytes)  # [B, D]
            
            # Decode this patch
            logits = self.decoder(patch, actual_patch_size)  # [B, actual_patch_size, V]
            all_logits.append(logits)
        
        # Concatenate all patches
        logits = torch.cat(all_logits, dim=1)  # [B, T, V]
        return logits
    
    def forward_phase2(
        self,
        bytes_seq: torch.Tensor,
        patch_size: int = 8,
    ) -> torch.Tensor:
        """Phase 2: hierarchical (encoder + patch-model + decoder).
        
        Same as phase 1, but insert patch-model between encoder and decoder.
        The patch-model maintains recurrent state across patches.
        
        Args:
            bytes_seq: [B, T] input bytes
            patch_size: N, bytes per patch
        Returns:
            logits: [B, T, V] reconstructed byte logits
        """
        B, T = bytes_seq.shape
        device = bytes_seq.device
        
        all_logits = []
        patch_state = torch.zeros(B, self.patch_state_dim, device=device)
        
        # Process in fixed-size patches
        for patch_start in range(0, T, patch_size):
            patch_end = min(patch_start + patch_size, T)
            actual_patch_size = patch_end - patch_start
            
            # Encode this patch
            patch_bytes = bytes_seq[:, patch_start:patch_end]
            patch = self.encoder(patch_bytes)  # [B, D]
            
            # Transform patch through patch-model
            transformed_patch, patch_state = self.patch_model(patch, patch_state)
            
            # Decode transformed patch
            logits = self.decoder(transformed_patch, actual_patch_size)
            all_logits.append(logits)
        
        # Concatenate all patches
        logits = torch.cat(all_logits, dim=1)  # [B, T, V]
        return logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    model = RNXPatchModel(byte_state_dim=8, patch_dim=8, patch_state_dim=16)
    print(f"RNN Patch Model params: {count_params(model):,}")
    print(f"  encoder:     {count_params(model.encoder):,}")
    print(f"  decoder:     {count_params(model.decoder):,}")
    print(f"  patch_model: {count_params(model.patch_model):,}")
    
    # Test forward pass
    B, T = 2, 64
    bytes_seq = torch.randint(1, 256, (B, T))
    
    # Phase 1
    logits = model.forward_phase1(bytes_seq, patch_size=8)
    print(f"\nPhase 1 (autoencoder):")
    print(f"  input:  {bytes_seq.shape}")
    print(f"  logits: {logits.shape}")
    
    # Phase 2
    logits = model.forward_phase2(bytes_seq, patch_size=8)
    print(f"\nPhase 2 (hierarchical):")
    print(f"  input:  {bytes_seq.shape}")
    print(f"  logits: {logits.shape}")
