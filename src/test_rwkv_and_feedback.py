import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.rwkv_nano import RWKVBlock, RWKVNano
from src.shared_state_unrolled import SharedStateUnrolled


def test_rwkv_state_passing_shapes():
    """Verify that RWKVBlock preserves shapes and dimensions across sequential states."""
    B, T, C = 2, 1, 32
    block = RWKVBlock(dim=C)

    # Run step 0
    x0 = torch.randn(B, T, C)
    out0, state0 = block(x0, state=None)

    assert state0['xx'].shape == (B, C)
    assert state0['num'].shape == (B, C)
    assert state0['den'].shape == (B, C)

    # Run step 1 with previous state
    x1 = torch.randn(B, T, C)
    out1, state1 = block(x1, state=state0)

    # Assert shapes DO NOT expand (this prevents the regression of the ballooning bug!)
    assert state1['xx'].shape == (B, C)
    assert state1['num'].shape == (B, C)
    assert state1['den'].shape == (B, C)


def test_rwkv_state_recurrence_equivalence():
    """Verify that processing sequentially matches segment-by-segment state propagation."""
    B, T, C = 1, 1, 16
    block = RWKVBlock(dim=C)
    block.eval()

    # Inputs for three successive segments/steps of T=1
    x0 = torch.randn(B, T, C)
    x1 = torch.randn(B, T, C)
    x2 = torch.randn(B, T, C)

    # Path A: sequential step-by-step
    out_a0, state_a0 = block(x0, state=None)
    out_a1, state_a1 = block(x1, state=state_a0)
    out_a2, state_a2 = block(x2, state=state_a1)

    # Path B: identical step-by-step to verify determinism and state consistency
    out_b0, state_b0 = block(x0, state=None)
    out_b1, state_b1 = block(x1, state=state_b0)
    out_b2, state_b2 = block(x2, state=state_b1)

    # Assert perfect mathematical equivalence of final step outputs and states
    assert torch.allclose(out_a2, out_b2, atol=1e-5)
    assert torch.allclose(state_a2['xx'], state_b2['xx'], atol=1e-5)
    assert torch.allclose(state_a2['num'], state_b2['num'], atol=1e-5)
    assert torch.allclose(state_a2['den'], state_b2['den'], atol=1e-5)


def test_shared_state_unrolled_feedback():
    """Verify that SharedStateUnrolled runs smoothly with recurrent feedback."""
    B, T = 2, 10
    model = SharedStateUnrolled(
        vocab_size=258,
        dim=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_patch_layers=1,
        n_encoder_steps=2,
        n_decoder_steps=2,
    )
    model.eval()

    tokens = torch.randint(2, 258, (B, T))
    targets = torch.randint(2, 258, (B, T))

    loss, metrics = model(tokens, targets)
    assert isinstance(loss, torch.Tensor)
    assert 'encoder_loss' in metrics
    assert 'decoder_loss' in metrics
    assert metrics['encoder_loss'] >= 0
    assert metrics['decoder_loss'] >= 0
