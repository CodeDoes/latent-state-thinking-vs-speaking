"""Token vs byte head model.

AdaptiveLoop but decoder head can be token vocab (BPE from byte_vocab).
For minimal test we use simple BPE built on top of byte vocab: 1024 tokens via learning from text.txt.
If byte_vocab BPE not available, fall back to byte 258 for both but simulate sparsity via strided loss.

For true token test, we build a small tokenizer using byte_vocab approach:
- First 256 bytes = byte values
- Additional merges learned? Simplified: we chunk bytes into groups of `bytes_per_token` (e.g., 4) for token target.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.adaptive_loop_model import AdaptiveLoopModel

class TokenByteHeadModel(nn.Module):
    def __init__(self, dim=64, patch_size=4, enc_layers=2, core_layers=2, dec_layers=2,
                 enc_max_loops=3, core_depth_loops=2, dec_max_loops=3,
                 vocab_mode="byte", token_vocab_size=1024, bytes_per_token=4):
        super().__init__()
        self.vocab_mode = vocab_mode
        self.bytes_per_token = bytes_per_token
        self.token_vocab_size = token_vocab_size

        # Base model with byte head
        self.base = AdaptiveLoopModel(
            dim=dim, patch_size=patch_size,
            enc_layers=enc_layers, core_layers=core_layers, dec_layers=dec_layers,
            enc_max_loops=enc_max_loops, core_depth_loops=core_depth_loops, dec_max_loops=dec_max_loops,
            dynamic_patch=False
        )

        if vocab_mode == "token":
            # Replace decoder head with token vocab head
            self.base.decoder.head = nn.Linear(dim, token_vocab_size)
            # Also need token embedding for detokenize? For training we still feed byte inputs,
            # target will be tokenized byte groups.
            # For simplicity, token = hashed byte n-gram bucket (not learned BPE) to keep deterministic.
            # This simulates sparsity: 1 token prediction per 4 bytes.
        # else byte mode unchanged

    def forward(self, input_ids_byte, targets_byte=None, targets_token=None):
        # input_ids_byte always bytes
        logits, info = self.base(input_ids_byte)  # [B,T,258 or token_vocab]

        if self.vocab_mode == "byte":
            # logits already byte logits, target is byte
            return logits, info
        else:
            # Token mode: we need to pool byte targets into token targets
            # logits shape [B, T, token_vocab] – we will downsample by bytes_per_token for loss
            # For training we predict token every bytes_per_token steps (sparse supervision)
            # Return raw logits but info includes token pooling
            return logits, info

    def byte_to_token_targets(self, byte_targets: torch.Tensor):
        """Convert [B,T] byte ids to [B,T//bpt] token ids via simple hashing bucket."""
        B,T = byte_targets.shape
        bpt = self.bytes_per_token
        trunc = (T // bpt) * bpt
        bt = byte_targets[:, :trunc].reshape(B, trunc//bpt, bpt)  # [B, num_tokens, bpt]
        # simple hash: sum bytes modulo vocab size offset by 256
        # Ensure within [0, token_vocab-1]
        token_ids = (bt.sum(dim=-1) % (self.token_vocab_size - 256)) + 256
        # Also need to keep byte targets for byte-equivalent eval? We return token ids
        return token_ids

    def token_logits_to_byte_loss(self, token_logits, byte_targets):
        """For token model, compute byte-equivalent loss by expanding token pred to bytes? Approx."""
        # This is only for logging, not training. We approximate by mapping token prediction's hashed bucket
        # back to byte via nearest? Simpler: we don't. Training uses token CE.
        # For eval we generate bytes via detokenizer that repeats token id %256 as byte guess.
        B,T,_ = token_logits.shape
        bpt = self.bytes_per_token
        # Take every bpt-th logit
        token_logits_down = token_logits[:, ::bpt, :]  # [B, T//bpt, vocab]
        token_targets = self.byte_to_token_targets(byte_targets)
        # Align lengths
        min_len = min(token_logits_down.shape[1], token_targets.shape[1])
        return token_logits_down[:,:min_len,:], token_targets[:,:min_len]
