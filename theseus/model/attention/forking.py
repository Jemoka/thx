"""
Attention module with forking support for Thoughtbubbles.
Extends RopeAttention with cumulative scores and token index tracking.
"""

from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.attention.rope import RopeAttention

ATTN_DTYPE = jnp.bfloat16


@jax.jit(static_argnums=0)
def causal_bias(max_block_size: int) -> jax.Array:
    neg = jnp.array(-jnp.inf, ATTN_DTYPE)
    m = jnp.tril(jnp.ones((max_block_size, max_block_size), dtype=jnp.bool_))
    return jnp.where(m, jnp.array(0, ATTN_DTYPE), neg)[None, None, :, :]


@jax.jit
def key_padding_bias(padding_mask: jax.Array) -> jax.Array:
    neg = jnp.array(-jnp.inf, ATTN_DTYPE)
    keep = padding_mask.astype(jnp.bool_)[:, None, None, :]
    return jnp.where(keep, jnp.array(0, ATTN_DTYPE), neg)


class ForkingAttention(RopeAttention):
    """Self-attention with forking support, extending RopeAttention."""

    block_size: int = field("architecture/block_size", default=512)
    max_block_size: int = field("architecture/max_block_size", default=1024)
    use_fork_channel: bool = field("architecture/use_fork_channel", default=True)

    def preprocess_qkv(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # q, k, v: (B, T, H, D)
        cumulative_scores: jax.Array = kwargs.get("cumulative_scores")  # type: ignore
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore

        if cumulative_scores is None or token_index is None:
            raise ValueError(
                "cumulative_scores and token_index must be provided for ForkingAttention"
            )

        token_counts = jnp.zeros(
            (*token_index.shape[:-1], self.block_size), dtype=token_index.dtype
        )
        token_counts = token_counts.at[
            jnp.arange(token_counts.shape[0])[:, None], token_index
        ].add(1)
        partial_rotations = jnp.cumsum(
            jnp.take_along_axis(1 / (token_counts + 1e-10), token_index, axis=-1),
            axis=-1,
        )
        q, k = self.rope(q, k, t=partial_rotations)
        if self.use_fork_channel:
            q = q.at[:, :, :, -1].set(jnp.ones_like(q[:, :, :, -1]))
            # cumulative_scores: (B, T) → broadcast to (B, T, H)
            k = k.at[:, :, :, -1].set(
                jnp.repeat(cumulative_scores[:, :, None], k.shape[2], axis=2)
                # such that we actually multiply by cumulative_scores in the attention computation
                # and not cumulative_scores^(1/sqrt(d_head)).
            )

        return q, k, v

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        # q, k, v: (B, T, H, D)
        cumulative_scores: jax.Array = kwargs.get("cumulative_scores")  # type: ignore
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore

        if cumulative_scores is None or token_index is None:
            raise ValueError(
                "cumulative_scores and token_index must be provided for ForkingAttention"
            )

        # Build attention bias/mask
        if mask is not None:
            # padding_mask = mask[:, None, None, :]
            padding_bias = key_padding_bias(mask)
            padding_bias = jnp.take_along_axis(
                padding_bias, token_index[:, None, None, :], axis=-1
            )
            attn_bias = (
                causal_bias(self.max_block_size)[:, :, : q.shape[1], : k.shape[1]]
                + padding_bias
            )
            boolean_mask = attn_bias == 0
            attn_bias = None
        else:
            attn_bias = causal_bias(self.max_block_size)[
                :, :, : q.shape[1], : k.shape[1]
            ]
            boolean_mask = None

        # Scale v by cumulative scores: (B, T, H, D) * (B, T) → (B, T, H, D)
        v = jnp.einsum("bthd,bt->bthd", v, jnp.exp(cumulative_scores))

        # dot_product_attention expects (B, T, H, D) — same as our convention
        q = q.astype(ATTN_DTYPE)
        k = k.astype(ATTN_DTYPE)
        v = v.astype(ATTN_DTYPE)

        if boolean_mask is not None:
            y = jax.nn.dot_product_attention(q, k, v, mask=boolean_mask)
        else:
            y = jax.nn.dot_product_attention(q, k, v, bias=attn_bias)

        return y

    @nn.compact
    def _cached_kv(
        self,
        k: jax.Array,
        v: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        cache_max_len: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Update KV cache if active. k, v: (B, T, H, D).

        Forking attention is a no-op cache: signature kept compatible with
        ``SelfAttention._cached_kv`` so the base ``__call__`` path doesn't
        break, but ``cache_max_len`` is unused here.

        Returns (k, v, cache_index_after_update) where cache_index is None
        when cache is not active (training mode).
        """
        del cache_max_len

        return k, v, None

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore

        if padding_mask is not None:
            padding_mask = jnp.take_along_axis(padding_mask, token_index, axis=-1)

        return super().postprocess_attn(y, padding_mask, deterministic, **kwargs)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        cache_max_len: Optional[int] = None,
        **kwargs: Any,
    ) -> jax.Array:
        del cache_max_len  # forking attention has no KV cache
        B, T, C = x.shape

        q, k, v = self.project(x)

        # For decode steps with cache, inject correct RoPE positions
        if self.has_variable("cache", "cache_index"):
            ci: Any = self.get_variable("cache", "cache_index")
            kwargs = {**kwargs, "positions": self._cached_positions(T, ci)}
        elif "positions" not in kwargs:
            kwargs = {
                **kwargs,
                "positions": self._positions_from_padding(T, padding_mask),
            }

        q, k, v = self.preprocess_qkv(q, k, v, **kwargs)

        y = self.attn(q, k, v, padding_mask, **kwargs)
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)

        y = y.reshape(B, T, C)
        y = self.output_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y
