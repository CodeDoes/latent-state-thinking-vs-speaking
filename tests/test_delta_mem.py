"""Unit and integration tests for Delta-Rule and RWKV-State Memory adapters."""

import unittest
import torch
import torch.nn as nn

from threads.delta_mem.delta_mem import DeltaRuleStateMemory, RWKVStateMemory
from threads.memory_growth.transformer_with_memory import (
    CausalSelfAttentionWithMemory,
    TransformerWithMemory,
)


class TestDeltaMem(unittest.TestCase):
    """Verifies shape correctness, gradient flow, and modes of operation."""

    def setUp(self):
        self.B, self.T = 2, 8
        self.hidden_size = 16
        self.query_size = 16
        self.key_size = 16
        self.value_size = 16
        self.output_size = 16

    def test_delta_rule_state_memory_shapes(self):
        mem = DeltaRuleStateMemory(
            hidden_size=self.hidden_size,
            query_size=self.query_size,
            key_size=self.key_size,
            value_size=self.value_size,
            output_size=self.output_size,
            rank=4,
            num_state_heads=2,
        )

        # 3D Batch shape
        x_3d = torch.randn(self.B, self.T, self.hidden_size)
        deltas = mem(x_3d)

        for head in ("q", "k", "v", "o"):
            self.assertIn(head, deltas)
            self.assertEqual(deltas[head].shape, (self.B, self.T, self.hidden_size))

        # 2D Flat sequence shape
        x_2d = torch.randn(self.T, self.hidden_size)
        deltas_2d = mem(x_2d)

        for head in ("q", "k", "v", "o"):
            self.assertIn(head, deltas_2d)
            self.assertEqual(deltas_2d[head].shape, (self.T, self.hidden_size))

    def test_rwkv_state_memory_shapes(self):
        mem = RWKVStateMemory(
            hidden_size=self.hidden_size,
            query_size=self.query_size,
            key_size=self.key_size,
            value_size=self.value_size,
            output_size=self.output_size,
        )

        # 3D Batch shape
        x_3d = torch.randn(self.B, self.T, self.hidden_size)
        deltas = mem(x_3d)

        for head in ("q", "k", "v", "o"):
            self.assertIn(head, deltas)
            self.assertEqual(deltas[head].shape, (self.B, self.T, self.hidden_size))

    def test_attention_integration(self):
        for mode in ("delta_rule", "rwkv7"):
            # Use "random" output_init so weights are non-zero and gradients flow
            attn = CausalSelfAttentionWithMemory(
                hidden_size=self.hidden_size,
                num_heads=2,
                rwkv_mem_enabled=True,
                rwkv_mem_mode=mode,
                rwkv_mem_rank=4,
                rwkv_mem_output_init="random",
            )

            x = torch.randn(self.B, self.T, self.hidden_size, requires_grad=True)
            out = attn(x)

            self.assertEqual(out.shape, (self.B, self.T, self.hidden_size))

            loss = out.sum()
            loss.backward()

            # Ensure used trainable parameters in the memory adapter have gradients
            for name, param in attn.named_parameters():
                if "rwkv_mem" in name:
                    # time_first is not used in the read-before-write shift, so ignore it
                    if "time_first" in name:
                        continue
                    self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                    self.assertGreater(
                        param.grad.abs().sum().item(),
                        0.0,
                        f"Parameter {name} gradient is zero",
                    )

    def test_full_transformer_with_memory(self):
        for mode in ("baseline", "delta_rule", "rwkv_state"):
            model = TransformerWithMemory(
                vocab_size=32,
                dim=self.hidden_size,
                num_layers=2,
                num_heads=2,
                rwkv_mem_enabled=(mode != "baseline"),
                rwkv_mem_mode="delta_rule" if mode == "delta_rule" else "rwkv7",
                rwkv_mem_rank=4,
            )

            tokens = torch.randint(1, 30, (self.B, self.T))
            logits = model(tokens)

            self.assertEqual(logits.shape, (self.B, self.T, 32))


if __name__ == "__main__":
    unittest.main()
