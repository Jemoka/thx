import jax.numpy as jnp
from typing import Tuple, Optional


class RotaryPosEncoding:
    """Unified RoPE supporting standard, Qwen, and NeoX variants.

    Differences between variants are controlled by constructor args:
    - base: 10000 (standard/NeoX) or 1e6 (Qwen)
    - partial_rotary_factor: 1.0 (standard/Qwen) or <1.0 (NeoX)
    """

    def __init__(
        self,
        d_model: int,
        base: int = 10000,
        seq_dim: int = 1,
        partial_rotary_factor: float = 1.0,
    ):
        self.d_model = d_model
        self.base = base
        self.seq_dim = seq_dim
        self.partial_rotary_factor = partial_rotary_factor

        rotary_dim = int(d_model * partial_rotary_factor)
        rotary_dim = max(2, rotary_dim)  # at least one pair
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (
            base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
        )
        self.inv_freq = inv_freq

    def rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        pos_offset: int = 0,
        t: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # q: (B, T, H, D), k: (B, T, K, D)
        b, tlen = q.shape[0], q.shape[self.seq_dim]

        # Compute positions
        if t is None:
            positions = jnp.arange(tlen, dtype=jnp.int32)[None, :]
        else:
            positions = t
            if positions.ndim == 1:
                positions = positions[None, :]
        positions = positions + pos_offset
        positions = jnp.broadcast_to(positions, (b, tlen))

        # Compute sin/cos: (B, T, rotary_dim)
        freqs = (
            positions[..., None].astype(self.inv_freq.dtype)
            * self.inv_freq[None, None, :]
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)[:, :, None, :]  # (B, T, 1, rotary_dim)
        sin = jnp.sin(emb)[:, :, None, :]

        # Split rotary and pass-through
        q_rot = q[..., : self.rotary_dim]
        k_rot = k[..., : self.rotary_dim]

        # Apply rotation
        q_out_rot = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
        k_out_rot = (k_rot * cos) + (self.rotate_half(k_rot) * sin)

        # Concatenate with pass-through if partial rotation
        if self.rotary_dim < self.d_model:
            q_out = jnp.concatenate([q_out_rot, q[..., self.rotary_dim :]], axis=-1)
            k_out = jnp.concatenate([k_out_rot, k[..., self.rotary_dim :]], axis=-1)
        else:
            q_out = q_out_rot
            k_out = k_out_rot

        return q_out, k_out
