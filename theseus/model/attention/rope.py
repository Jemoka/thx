"""
Attention module with Rotary Positional Encoding (RoPE).

Inherits KV cache support from SelfAttention. During cached decode,
the base class automatically injects the correct absolute position
into kwargs["positions"] so RoPE applies the right rotation.
"""

import jax
from typing import Any, Tuple

from theseus.model.layers.rope import RotaryPosEncoding
from theseus.model.attention.base import SelfAttention


class RopeAttention(SelfAttention):
    def setup(self) -> None:
        super().setup()
        # seq_dim=1 for (B, T, H, D) convention
        self.rope = RotaryPosEncoding(self.head_dim, seq_dim=1)

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # positions kwarg is injected by base __call__ during cached decode
        q, k = self.rope(q, k, t=kwargs.get("positions"))
        return q, k, v
