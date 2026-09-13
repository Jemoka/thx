"""Qwen 3.5 hybrid decoder block.

Each layer's ``layer_type`` selects the token mixer:
  * ``"full_attention"``: ``GroupedSelfAttention`` with Q-output gate and
    per-head Q/K RMSNorm.
  * ``"linear_attention"``: ``GatedDeltaNet``.

The MLP is set by the model — subclasses override ``_make_mlp`` to swap in a
MoE for the 35B-A3B variant.
"""

from typing import Any, List, Optional, Type

import jax

from theseus.base.axis import ShardingPlan
from theseus.config import configure
from theseus.model.attention.gated_delta import GatedDeltaNet
from theseus.model.attention.grouped import GroupedSelfAttention
from theseus.model.layers import QwenMLP, RMSNorm
from theseus.model.moe.shared import SharedGatedMoE
from theseus.model.module import Module


class Qwen3_5DecoderBlock(Module):
    """Dense decoder block. Layer type is fixed at construction time."""

    layer_type: str = "full_attention"

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm, GroupedSelfAttention, GatedDeltaNet, QwenMLP]

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    def _make_mlp(self) -> Any:
        return configure(QwenMLP)

    def flops(self, seq: int) -> float:
        return float(self.attn.flops(seq) + self.mlp.flops(seq))

    def setup(self) -> None:
        self.rms_1 = configure(RMSNorm, centered=True)
        if self.layer_type == "linear_attention":
            self.attn = configure(GatedDeltaNet)
        else:
            self.attn = configure(
                GroupedSelfAttention, q_output_gate=True, qk_head_norm=True
            )
        self.rms_2 = configure(RMSNorm, centered=True)
        self.mlp = self._make_mlp()

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        positions: Optional[jax.Array] = None,
        cache_max_len: Optional[int] = None,
    ) -> jax.Array:
        h = self.rms_1(x)
        h = self.attn(
            h,
            padding_mask=padding_mask,
            deterministic=deterministic,
            positions=positions,
            cache_max_len=cache_max_len,
        )
        x = x + h

        h = self.rms_2(x)
        h = self.mlp(h, deterministic=deterministic)
        x = x + h
        return x


class Qwen3_5MoEDecoderBlock(Qwen3_5DecoderBlock):
    """Same hybrid attention layout as the dense block; MoE FFN instead of MLP."""

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm, GroupedSelfAttention, GatedDeltaNet, SharedGatedMoE]

    def _make_mlp(self) -> Any:
        return configure(SharedGatedMoE)
