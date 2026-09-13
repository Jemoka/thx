import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Type, Any, Optional

from theseus.base.axis import ShardingPlan
from theseus.config import field
from theseus.model.module import Module


class RMSNorm(Module):
    """Root-mean-square layer norm.

    ``centered=True`` switches to the Qwen 3.5 / OLMo-2 convention:
    multiplier is ``1 + weight`` with weight init at 0 (numerically still
    centered at 1, but lets HF checkpoints round-trip).
    """

    ndim: int = field("architecture/n_embd", default=2048)
    eps: float = field("architecture/rms_norm_eps", default=1e-6)
    centered: bool = False

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        init = nn.initializers.zeros if self.centered else nn.initializers.ones
        weight = self.param("weight", init, (self.ndim,), self._param_dtype)
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)
        scale = (
            (1.0 + weight.astype(jnp.float32))
            if self.centered
            else weight.astype(jnp.float32)
        )
        return (x_norm * scale).astype(x.dtype)


class RMSNormGated(Module):
    """RMSNorm with a multiplicative ``silu(gate)`` after the weight.

    Used inside the gated-delta-net token mixer (Qwen 3.5 linear-attention
    layers). Weight is initialized to ones — matches HF ``Qwen3_5RMSNormGated``.
    """

    ndim: int = field("architecture/n_embd", default=2048)
    eps: float = field("architecture/rms_norm_eps", default=1e-6)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    @nn.compact
    def __call__(self, x: jax.Array, gate: Optional[jax.Array] = None) -> jax.Array:
        weight = self.param(
            "weight", nn.initializers.ones, (self.ndim,), self._param_dtype
        )
        out_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)
        x_norm = weight.astype(jnp.float32) * x_norm
        if gate is not None:
            x_norm = x_norm * jax.nn.silu(gate.astype(jnp.float32))
        return x_norm.astype(out_dtype)
