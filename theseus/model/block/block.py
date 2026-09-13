"""
Your most very basic indeed transformer Block
"""

from typing import Any, Dict, List, Optional, Type, cast

import jax
import jax.numpy as jnp

from theseus.model.attention import SelfAttention, RopeAttention
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.layers.mlp import MLP
from theseus.model.module import Module
from theseus.model.utils import mean_nonpadding, state_cosine

from theseus.config import configure, field


class Block(Module):
    rope: bool = field("architecture/rope", default=True)
    instrument_components: bool = field(
        "architecture/instrumentation/components", default=False
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, SelfAttention, RopeAttention, MLP]

    def flops(self, seq: int) -> float:
        return float(self.attn.flops(seq) + self.mlp.flops(seq))

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        if self.rope:
            self.attn = configure(RopeAttention)
        else:
            self.attn = configure(SelfAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(MLP)

    def __call__(self, x: jax.Array, **kwargs: Any) -> jax.Array:
        x_attn = x + self.attn(self.ln_1(x), **kwargs)
        x_out = x_attn + self.mlp(self.ln_2(x_attn))
        if self.instrument_components:
            self.measure(x, x_attn, x_out, kwargs)
        return cast(jax.Array, x_out)

    def measure(
        self,
        residual: jax.Array,
        after_attention: jax.Array,
        output: jax.Array,
        kwargs: Dict[str, Any],
    ) -> None:
        """Measure each sublayer's write to its receiving residual."""

        ####### component diagnostics #######

        effective_depth = cast(Optional[int], kwargs.get("effective_depth"))
        if effective_depth is None:
            return
        padding_mask = cast(Optional[jax.Array], kwargs.get("padding_mask"))

        for name, update, update_residual in (
            ("attention", after_attention - residual, residual),
            ("mlp", output - after_attention, after_attention),
        ):
            update_f32 = update.astype(jnp.float32)
            update_residual_f32 = update_residual.astype(jnp.float32)
            update_rms = jnp.sqrt(jnp.mean(update_f32**2, axis=-1))
            update_residual_rms = jnp.sqrt(jnp.mean(update_residual_f32**2, axis=-1))
            self.sow(
                "scalars",
                f"depth/{effective_depth}/{name}_update_rms_norm",
                mean_nonpadding(update_rms, padding_mask),
            )
            self.sow(
                "scalars",
                f"depth/{effective_depth}/{name}_update_to_residual_norm",
                mean_nonpadding(
                    update_rms / jnp.maximum(update_residual_rms, 1e-8),
                    padding_mask,
                ),
            )
            self.sow(
                "scalars",
                f"depth/{effective_depth}/{name}_update_residual_cosine",
                mean_nonpadding(
                    state_cosine(update_f32, update_residual_f32),
                    padding_mask,
                ),
            )

        ####### end component diagnostics #######
