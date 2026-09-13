"""Multi-modal RoPE (M-RoPE).

For text-only inputs (position_ids of shape ``(B, T)``) this is bit-exact to
:class:`RotaryPosEncoding` — the three (T, H, W) frequency channels all carry
the same positions, and the interleaved selection picks the same value at
every index. The module is broken out so the vision path can compose it later.
"""

from typing import Optional, Tuple, Sequence

import jax.numpy as jnp

from theseus.model.layers.rope import RotaryPosEncoding


class MRotaryPosEncoding(RotaryPosEncoding):
    """RoPE with optional 3-channel (T, H, W) position ids.

    Pass ``positions`` as ``(B, T)`` for text-only; as ``(3, B, T)`` for
    multimodal — the channels are interleaved per ``mrope_section``.
    """

    def __init__(
        self,
        d_model: int,
        base: int = 10000,
        seq_dim: int = 1,
        partial_rotary_factor: float = 1.0,
        mrope_section: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            base=base,
            seq_dim=seq_dim,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.mrope_section = tuple(mrope_section) if mrope_section else None

    def _interleave_mrope(self, freqs: jnp.ndarray) -> jnp.ndarray:
        """``freqs``: (3, B, T, rotary_dim/2) -> (B, T, rotary_dim/2).

        Frequency slots at positions ``offset::3`` (for offset=1,2) within the
        first ``mrope_section[dim]*3`` slots are overwritten with the
        corresponding channel's frequencies — matches HF Qwen 3.5.
        """
        out = freqs[0]
        if not self.mrope_section:
            return out
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            idx = jnp.arange(offset, length, 3)
            out = out.at[..., idx].set(freqs[dim_idx][..., idx])
        return out

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        pos_offset: int = 0,
        t: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if t is None or t.ndim < 3:
            return super().__call__(q, k, pos_offset=pos_offset, t=t)

        # t is (3, B, T): multimodal path.
        positions = (t + pos_offset).astype(self.inv_freq.dtype)
        freqs = positions[..., None] * self.inv_freq[None, None, None, :]
        freqs = self._interleave_mrope(freqs)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)[:, :, None, :]
        sin = jnp.sin(emb)[:, :, None, :]

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q_out_rot = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
        k_out_rot = (k_rot * cos) + (self.rotate_half(k_rot) * sin)
        if self.rotary_dim < self.d_model:
            q_out = jnp.concatenate([q_out_rot, q_pass], axis=-1)
            k_out = jnp.concatenate([k_out_rot, k_pass], axis=-1)
        else:
            q_out, k_out = q_out_rot, k_out_rot
        return q_out, k_out
