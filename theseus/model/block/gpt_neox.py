from typing import Optional, List, Type, Any

import jax
import flax.linen as nn

from theseus.base.axis import ShardingPlan
from theseus.config import field, configure
from theseus.model.module import Module
from theseus.model.layers import LayerNorm, NeoXMLP
from theseus.model.attention.grouped import GroupedSelfAttention


class GPTNeoXDecoderBlock(Module):
    dropout: float = field("architecture/dropout", default=0.0)
    use_parallel_residual: bool = field(
        "architecture/use_parallel_residual", default=True
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, GroupedSelfAttention, NeoXMLP]

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    def flops(self, seq: int) -> float:
        return float(self.attn.flops(seq) + self.mlp.flops(seq))

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        self.attn = configure(GroupedSelfAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(NeoXMLP)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        positions: Optional[jax.Array] = None,
    ) -> jax.Array:
        attn_out = self.attn(
            self.ln_1(x),
            padding_mask=padding_mask,
            deterministic=deterministic,
            sliding=False,
            positions=positions,
        )
        attn_out = self.dropout_layer(attn_out, deterministic=deterministic)

        if self.use_parallel_residual:
            mlp_out = self.mlp(self.ln_2(x), deterministic=deterministic)
            mlp_out = self.dropout_layer(mlp_out, deterministic=deterministic)
            x = x + attn_out + mlp_out
        else:
            x = x + attn_out
            mlp_out = self.mlp(self.ln_2(x), deterministic=deterministic)
            mlp_out = self.dropout_layer(mlp_out, deterministic=deterministic)
            x = x + mlp_out
        return x
