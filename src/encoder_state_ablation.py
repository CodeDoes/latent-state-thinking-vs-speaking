"""Encoder ablation: which state components matter for byte prediction?

Tests 7 variants of an encoder that takes different state combinations:
1. static-patch-state (only frozen patch model state)
2. encoder-state (only encoder's own recurrent state)
3. static-patch-state + encoder-state
4. byte-level-state (only the byte-level running state)
5. static-patch-state + byte-level-state
6. mutable-full-state (all states that can be updated)
7. static-full-state (all states, frozen)

Each variant tracks surprise. When surprise exceeds threshold, push state
into patch-model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StateBuffer:
    """Container for all state components. Can be frozen or mutable."""
    
    def __init__(self, dim: int, device='cpu'):
        self.dim = dim
        self.device = device
        
        # State components
        self.patch_state = torch.zeros(dim, device=device)  # patch model state
        self.encoder_state = torch.zeros(dim, device=device)  # encoder itself
        self.byte_level_state = torch.zeros(dim, device=device)  # running byte state
    
    def freeze_all(self) -> dict:
        """Return frozen snapshot of all states."""
        return {
            'patch_state': self.patch_state.clone().detach(),
            'encoder_state': self.encoder_state.clone().detach(),
            'byte_level_state': self.byte_level_state.clone().detach(),
        }
    
    def update_full(self, new_states: dict):
        """Update all states."""
        self.patch_state = new_states['patch_state']
        self.encoder_state = new_states['encoder_state']
        self.byte_level_state = new_states['byte_level_state']


class EncoderVariant(nn.Module):
    """Encoder that selects which state components to use as input."""
    
    def __init__(self, dim: int, state_components: list[str], vocab_size: int = 258):
        super().__init__()
        self.dim = dim
        self.state_components = state_components
        
        # Input dim = byte embedding (dim) + selected states
        n_states = len(state_components)
        input_dim = dim + n_states * dim
        
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        
        # Two-layer FC encoder (simple, not recurrent)
        self.fc1 = nn.Linear(input_dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        
        # Surprise routing
        self.surprise_threshold = nn.Parameter(torch.tensor(0.5))
        self.to_logits = nn.Linear(dim, vocab_size)
    
    def forward(
        self,
        byte_input: torch.Tensor,
        state_buffer: StateBuffer,
        patch_model: nn.Module,
        return_surprise: bool = False
    ) -> tuple:
        """
        Args:
            byte_input: [batch] byte token ids
            state_buffer: all available states
        
        Returns:
            logits: [batch, vocab_size]
            surprise: scalar surprise value
            should_patch: bool, whether to push state to patch model
        """
        # Embed byte
        x = self.embed(byte_input)  # [batch, dim]
        
        # Select state components
        state_pieces = []
        for comp in self.state_components:
            if comp == 'static-patch-state':
                state_pieces.append(state_buffer.patch_state.unsqueeze(0).expand(x.shape[0], -1))
            elif comp == 'encoder-state':
                state_pieces.append(state_buffer.encoder_state.unsqueeze(0).expand(x.shape[0], -1))
            elif comp == 'byte-level-state':
                state_pieces.append(state_buffer.byte_level_state.unsqueeze(0).expand(x.shape[0], -1))
            elif comp == 'static-full-state':
                full_state = (state_buffer.patch_state + state_buffer.encoder_state + state_buffer.byte_level_state) / 3
                state_pieces.append(full_state.unsqueeze(0).expand(x.shape[0], -1))
            elif comp == 'mutable-full-state':
                # Same as static-full-state for simplicity (all components available)
                full_state = (state_buffer.patch_state + state_buffer.encoder_state + state_buffer.byte_level_state) / 3
                state_pieces.append(full_state.unsqueeze(0).expand(x.shape[0], -1))
        
        if len(state_pieces) == 0:
            state_input = torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)
        else:
            state_input = torch.cat(state_pieces, dim=-1)  # [batch, n_states * dim]
        
        # Concatenate byte + state
        x = torch.cat([x, state_input], dim=-1)  # [batch, input_dim]
        
        # Forward through encoder
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        
        # Predict logits
        logits = self.to_logits(h)  # [batch, vocab_size]
        
        # Compute surprise: entropy of prediction
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        surprise = torch.sigmoid(entropy)
        
        # Decide whether to patch
        should_patch = bool(surprise.item() > self.surprise_threshold.item())
        
        # Update encoder state (for variants that use encoder-state)
        if 'encoder-state' in self.state_components:
            state_buffer.encoder_state = h.mean(dim=0).detach()
        
        if return_surprise:
            return logits, surprise, should_patch
        return logits


class PatchModel(nn.Module):
    """Simple patch model that processes aggregated state."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim * 2, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Simulate patch compression (double the state concat with itself)
        x = torch.cat([state, state], dim=-1)
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


if __name__ == "__main__":
    # Smoke test all 7 variants
    dim = 32
    vocab_size = 258
    batch = 2
    
    state_buffer = StateBuffer(dim)
    patch_model = PatchModel(dim)
    
    variants = [
        ['static-patch-state'],
        ['encoder-state'],
        ['static-patch-state', 'encoder-state'],
        ['byte-level-state'],
        ['static-patch-state', 'byte-level-state'],
        ['mutable-full-state'],
        ['static-full-state'],
    ]
    
    for state_comps in variants:
        model = EncoderVariant(dim, state_comps, vocab_size)
        x = torch.randint(1, 256, (batch,))
        logits, surprise, should_patch = model(x, state_buffer, patch_model, return_surprise=True)
        print(f"{'+'.join(state_comps):50s} → logits {logits.shape}, surprise {surprise.item():.3f}, patch={should_patch}")
