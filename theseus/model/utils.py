from typing import Optional

import jax
import jax.numpy as jnp

####### diagnostic utilities #######


def mean_nonpadding(values: jax.Array, padding_mask: Optional[jax.Array]) -> jax.Array:
    """Average one scalar per token, excluding padded positions."""
    if padding_mask is None:
        return jnp.mean(values)
    mask = padding_mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1)


def state_cosine(left: jax.Array, right: jax.Array) -> jax.Array:
    """Compute per-token cosine similarity between two residual states."""
    left_f32 = left.astype(jnp.float32)
    right_f32 = right.astype(jnp.float32)
    left_rms = jnp.sqrt(jnp.mean(left_f32**2, axis=-1))
    right_rms = jnp.sqrt(jnp.mean(right_f32**2, axis=-1))
    return jnp.mean(left_f32 * right_f32, axis=-1) / jnp.maximum(
        left_rms * right_rms, 1e-8
    )


####### end diagnostic utilities #######
