from typing import Optional, List, Type, Any

import jax

from theseus.base.axis import ShardingPlan
from theseus.config import configure
from theseus.model.module import Module
from theseus.model.layers import RMSNorm, QwenMLP
from theseus.model.attention.grouped import GroupedSelfAttention


class QwenDecoderBlock(Module):
    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm, GroupedSelfAttention, QwenMLP]

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    def flops(self, seq: int) -> float:
        return float(self.attn.flops(seq) + self.mlp.flops(seq))

    def setup(self) -> None:
        self.rms_1 = configure(RMSNorm)
        self.attn = configure(GroupedSelfAttention)
        self.rms_2 = configure(RMSNorm)
        self.mlp = configure(QwenMLP)

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        sliding: bool = False,
        positions: Optional[jax.Array] = None,
        cache_max_len: Optional[int] = None,
    ) -> jax.Array:
        h = self.rms_1(x)
        h = self.attn(
            h,
            padding_mask=padding_mask,
            deterministic=deterministic,
            sliding=sliding,
            positions=positions,
            cache_max_len=cache_max_len,
        )
        x = x + h

        h = self.rms_2(x)
        h = self.mlp(h, deterministic=deterministic)
        x = x + h
        return x
