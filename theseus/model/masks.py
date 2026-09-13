import jax.numpy as jnp
from typing import Optional


def causal_mask(seq_len: int) -> jnp.ndarray:
    """Boolean causal mask (1,1,T,T), True=keep."""
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def sliding_window_mask(seq_len: int, window: int) -> jnp.ndarray:
    """Sliding window causal mask (1,1,T,T).

    Query at row ``i`` attends to keys ``j`` with ``i - window < j <= i``
    (i.e. the last ``window`` past tokens, inclusive of self). The distance is query minus key, so future tokens are excluded.
    """
    idx = jnp.arange(seq_len)
    dist = idx[:, None] - idx[None, :]  # query - key
    mask = (dist >= 0) & (dist < window)
    return mask[None, None, :, :]


def combine_padding(mask: jnp.ndarray, padding: Optional[jnp.ndarray]) -> jnp.ndarray:
    if padding is None:
        return mask
    return mask & padding[:, None, None, :]


def cache_mask(max_length: int, cache_index: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask for KV cache: attend only to positions < cache_index.

    Returns (1, 1, 1, max_length) bool mask, True=keep.
    """
    return (jnp.arange(max_length) < cache_index)[None, None, None, :]
