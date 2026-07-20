"""Encoder-Patcher-Decoder with dual surprise-router loops.

Architecture:
    byte-input → encoder → surprise-router → byte-input (loop if not done)
    surprise-router → patcher → decoder → surprise-router-2 → byte-output
    surprise-router-2 → decoder (loop if not confident)

The encoder can iterate on the input until surprise-router says "state is
good enough." The decoder can iterate on the output until surprise-router-2
says "output is confident enough."

Shared state flows: encoder → patcher → decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from threads.unsorted.simple_rnn_receptance import SimpleRNNReceptance


class SurpriseRouter(nn.Module):
    """Decides when the state/output is 'good enough' to move on.
    
    Outputs a scalar in [0, 1] per position. High = surprised (need more
    computation), Low = confident (move on).
    
    The loop continues while mean(surprise) > threshold.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns surprise score [batch, seq_len] in [0, 1]."""
        return torch.sigmoid(self.proj(x)).squeeze(-1)


class Encoder(nn.Module):
    """Byte-level encoder with surprise-router loop.
    
    Processes byte input, can loop multiple times until surprise-router
    says the state is good enough.
    """
    
    def __init__(self, dim: int, max_loops: int = 4):
        super().__init__()
        self.embed = nn.Embedding(258, dim, padding_idx=0)
        self.rnn = SimpleRNNReceptance(input_dim=dim, hidden_dim=dim)
        self.surprise_router = SurpriseRouter(dim)
        self.max_loops = max_loops
        self.ln = nn.LayerNorm(dim)
    
    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            tokens: [batch, seq_len] byte token ids
            state: previous hidden state [batch, dim]
        
        Returns:
            output: [batch, seq_len, dim] encoder output
            final_state: [batch, dim] final hidden state
            surprise: [batch, seq_len] surprise signal from loop exit
            info: dict with loop_count, surprise_history
        """
        x = self.embed(tokens)  # [batch, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = torch.zeros(batch_size, self.rnn.hidden_dim, device=x.device, dtype=x.dtype)
        
        surprise_history = []
        h = state
        
        for loop_idx in range(self.max_loops):
            # Process through RNN
            out, h, receptance = self.rnn(x, h)
            out = self.ln(out)
            
            # Check surprise router
            surprise = self.surprise_router(out)  # [batch, seq_len]
            surprise_history.append(surprise.detach())
            
            # If not surprised, exit loop
            if surprise.mean() < 0.5:  # threshold
                break
            
            # Loop: feed output back as input for next iteration
            x = out
        
        info = {
            "loop_count": loop_idx + 1,
            "surprise_history": surprise_history,
        }
        
        return out, h, surprise, info


class Patcher(nn.Module):
    """Compresses byte-level representations into patch-level.
    
    Mean-pools bytes within fixed-size windows, then processes with RNN.
    """
    
    def __init__(self, dim: int, patch_size: int = 4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.rnn = SimpleRNNReceptance(input_dim=dim, hidden_dim=dim)
        self.gate = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """
        Args:
            x: [batch, seq_len, dim] byte-level representations
            state: previous patch state [batch, dim]
        
        Returns:
            patches: [batch, n_patches, dim] patch-level representations
            final_state: [batch, dim] final patch state
        """
        batch_size, seq_len, dim = x.shape
        k = self.patch_size
        
        # Truncate to multiple of patch_size
        seq_trunc = (seq_len // k) * k
        if seq_trunc == 0:
            return torch.zeros(batch_size, 0, dim, device=x.device, dtype=x.dtype), state
        
        # Mean-pool bytes within each patch window
        x_trunc = x[:, :seq_trunc, :]
        patches = x_trunc.view(batch_size, seq_trunc // k, k, dim).mean(dim=2)
        
        # Process patches through RNN
        if state is None:
            state = torch.zeros(batch_size, self.rnn.hidden_dim, device=x.device, dtype=x.dtype)
        
        patch_out, patch_state, _ = self.rnn(patches, state)
        patch_out = patch_out * torch.sigmoid(self.gate)
        
        return patch_out, patch_state


class Decoder(nn.Module):
    """Byte-level decoder with surprise-router-2 loop.
    
    Takes patch-level context, can loop to refine byte predictions until
    surprise-router-2 says output is confident enough.
    """
    
    def __init__(self, dim: int, max_loops: int = 4):
        super().__init__()
        self.rnn = SimpleRNNReceptance(input_dim=dim, hidden_dim=dim)
        self.surprise_router_2 = SurpriseRouter(dim)
        self.max_loops = max_loops
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 258)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        patch_out: torch.Tensor,
        patch_size: int,
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            encoder_out: [batch, seq_len, dim] from encoder
            patch_out: [batch, n_patches, dim] from patcher
            patch_size: bytes per patch (for broadcasting)
            state: previous decoder state [batch, dim]
        
        Returns:
            logits: [batch, seq_len, vocab_size] byte predictions
            final_state: [batch, dim] final decoder state
            surprise: [batch, seq_len] surprise signal from loop exit
            info: dict with loop_count, surprise_history
        """
        batch_size, seq_len, dim = encoder_out.shape
        
        # Broadcast patches to byte positions
        n_patches = patch_out.shape[1]
        patch_broadcast = patch_out.repeat_interleave(patch_size, dim=1)
        if patch_broadcast.shape[1] > seq_len:
            patch_broadcast = patch_broadcast[:, :seq_len, :]
        elif patch_broadcast.shape[1] < seq_len:
            pad = torch.zeros(batch_size, seq_len - patch_broadcast.shape[1], dim,
                            device=patch_out.device, dtype=patch_out.dtype)
            patch_broadcast = torch.cat([patch_broadcast, pad], dim=1)
        
        # Combine encoder and patch context
        x = encoder_out + patch_broadcast
        
        if state is None:
            state = torch.zeros(batch_size, self.rnn.hidden_dim, device=x.device, dtype=x.dtype)
        
        surprise_history = []
        h = state
        
        for loop_idx in range(self.max_loops):
            # Process through RNN
            out, h, receptance = self.rnn(x, h)
            out = self.ln(out)
            
            # Check surprise-router-2
            surprise = self.surprise_router_2(out)  # [batch, seq_len]
            surprise_history.append(surprise.detach())
            
            # If confident, exit loop
            if surprise.mean() < 0.5:  # threshold
                break
            
            # Loop: feed output back as input for refinement
            x = out
        
        logits = self.head(out)
        
        info = {
            "loop_count": loop_idx + 1,
            "surprise_history": surprise_history,
        }
        
        return logits, h, surprise, info


class EncoderPatcherDecoder(nn.Module):
    """Full architecture: encoder with loop → patcher → decoder with loop.
    
    Shared state flows through all three components.
    """
    
    def __init__(self, dim: int = 64, patch_size: int = 4, max_loops: int = 4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.max_loops = max_loops
        
        self.encoder = Encoder(dim, max_loops)
        self.patcher = Patcher(dim, patch_size)
        self.decoder = Decoder(dim, max_loops)
    
    def forward(
        self,
        tokens: torch.Tensor,
        encoder_state: torch.Tensor | None = None,
        patcher_state: torch.Tensor | None = None,
        decoder_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            tokens: [batch, seq_len] byte token ids
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            info: dict with encoder/patcher/decoder info
        """
        # Encoder with surprise-router loop
        encoder_out, encoder_state, encoder_surprise, encoder_info = self.encoder(
            tokens, encoder_state
        )
        
        # Patcher compresses byte-level to patch-level
        patch_out, patcher_state = self.patcher(encoder_out, patcher_state)
        
        # Decoder with surprise-router-2 loop
        logits, decoder_state, decoder_surprise, decoder_info = self.decoder(
            encoder_out, patch_out, self.patch_size, decoder_state
        )
        
        info = {
            "encoder": encoder_info,
            "patcher": {"n_patches": patch_out.shape[1]},
            "decoder": decoder_info,
            "encoder_surprise": encoder_surprise.detach(),
            "decoder_surprise": decoder_surprise.detach(),
        }
        
        return logits, info


if __name__ == "__main__":
    # Smoke test
    batch_size, seq_len = 2, 32
    model = EncoderPatcherDecoder(dim=64, patch_size=4, max_loops=4)
    
    tokens = torch.randint(1, 256, (batch_size, seq_len))
    logits, info = model(tokens)
    
    print(f"Input tokens: {tokens.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Encoder loops: {info['encoder']['loop_count']}")
    print(f"Decoder loops: {info['decoder']['loop_count']}")
    print(f"Patches: {info['patcher']['n_patches']}")
    print(f"Encoder surprise: {info['encoder_surprise'].mean():.3f}")
    print(f"Decoder surprise: {info['decoder_surprise'].mean():.3f}")
    
    # Gradient check
    loss = logits.sum()
    loss.backward()
    
    n_tensors = sum(1 for _ in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\nTensors with gradients: {n_grad}/{n_tensors}")
    print("Gradient flow: OK" if n_grad == n_tensors else "Gradient flow: ISSUE")
