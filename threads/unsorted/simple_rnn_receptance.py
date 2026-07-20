"""Simplified RNN with receptance gate for surprise-based patch boundaries.

Minimal RNN cell with a receptance gate that can be extracted as a
"surprise" signal for dynamic patch boundary detection.

Architecture:
- Simple RNN: h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)
- Receptance gate: r_t = sigmoid(W_r @ h_{t-1} + W_xr @ x_t)
- Gated update: h_t = r_t * h_{t-1} + (1 - r_t) * tanh(...)

The receptance signal r_t ∈ [0, 1] indicates:
- Low r_t (close to 0): "keep old state, input is predictable"
- High r_t (close to 1): "accept new input, this is surprising"

This receptance signal can be used to trigger patch boundaries:
when r_t exceeds a threshold, the state is "full" and should be patched.

Shared state: encoder, patcher, and decoder all operate on the same
state vector. The receptance signal is extracted at each step to
determine when to patch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRNNReceptance(nn.Module):
    """Simplified RNN with receptance gate.
    
    Args:
        input_dim: dimension of input embeddings
        hidden_dim: dimension of hidden state
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Standard RNN weights
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Receptance gate weights
        self.W_r_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_r_x = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Output projection (hidden -> input_dim for next layer or prediction)
        self.W_out = nn.Linear(hidden_dim, input_dim, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor [batch, seq_len, input_dim]
            h_prev: previous hidden state [batch, hidden_dim] or None
        
        Returns:
            output: [batch, seq_len, input_dim]
            h_final: final hidden state [batch, hidden_dim]
            receptance: surprise signal [batch, seq_len] (mean over hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        outputs = []
        receptances = []
        h_t = h_prev
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_dim]
            
            # Compute receptance gate (surprise signal)
            r_t = torch.sigmoid(self.W_r_h(h_t) + self.W_r_x(x_t))  # [batch, hidden_dim]
            
            # Compute candidate hidden state
            h_candidate = torch.tanh(self.W_h(h_t) + self.W_x(x_t))  # [batch, hidden_dim]
            
            # Gated update: blend old state with new information
            h_t = r_t * h_t + (1 - r_t) * h_candidate  # [batch, hidden_dim]
            
            # Project to output dimension
            out_t = self.W_out(h_t)  # [batch, input_dim]
            
            outputs.append(out_t)
            # Receptance signal: mean over hidden dimensions (scalar per position)
            receptances.append(r_t.mean(dim=-1))  # [batch]
        
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, input_dim]
        receptance = torch.stack(receptances, dim=1)  # [batch, seq_len]
        
        return output, h_t, receptance


class SimpleRNNReceptanceLayer(nn.Module):
    """Stackable layer with layer norm and residual connection.
    
    Args:
        dim: dimension (input = output = hidden)
        n_layers: number of stacked RNN layers
    """
    
    def __init__(self, dim: int, n_layers: int = 1):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            SimpleRNNReceptance(input_dim=dim, hidden_dim=dim)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        states: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: input [batch, seq_len, dim]
            states: list of previous hidden states for each layer
        
        Returns:
            output: [batch, seq_len, dim]
            final_states: list of final hidden states
            receptance: aggregated surprise signal [batch, seq_len]
        """
        batch_size = x.shape[0]
        if states is None:
            states = [torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype) 
                     for _ in range(self.n_layers)]
        
        receptance_sum = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        final_states = []
        
        h = x
        for layer, ln, h_prev in zip(self.layers, self.layer_norms, states):
            out, h_final, r = layer(ln(h), h_prev)
            h = h + out  # residual connection
            receptance_sum += r
            final_states.append(h_final)
        
        # Average receptance across layers
        receptance = receptance_sum / self.n_layers
        
        return h, final_states, receptance


if __name__ == "__main__":
    # Smoke test
    batch_size, seq_len, dim = 2, 10, 64
    model = SimpleRNNReceptanceLayer(dim=dim, n_layers=2)
    
    x = torch.randn(batch_size, seq_len, dim)
    output, states, receptance = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Receptance shape: {receptance.shape}")
    print(f"Receptance range: [{receptance.min():.3f}, {receptance.max():.3f}]")
    print(f"Receptance mean: {receptance.mean():.3f}")
    print(f"Number of states: {len(states)}")
    print(f"State shapes: {[s.shape for s in states]}")
    
    # Test gradient flow
    loss = output.sum() + receptance.sum()
    loss.backward()
    
    n_params = sum(p.numel() for p in model.parameters())
    n_tensors = sum(1 for _ in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\nParameters: {n_params} scalars across {n_tensors} tensors")
    print(f"Tensors with gradients: {n_grad}/{n_tensors}")
    print("Gradient flow: OK" if n_grad == n_tensors else "Gradient flow: ISSUE")
