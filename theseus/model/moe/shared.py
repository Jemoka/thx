"""Top-k routed MoE with a sigmoid-gated shared expert (Qwen 3.5 MoE).

Differences vs. :class:`theseus.model.moe.base.MoE`:
  * Experts are gated SiLU MLPs (``QwenMLP``) sized by
    ``architecture/moe_intermediate_size`` instead of the layer-wide
    ``architecture/intermediate_size``.
  * Routing is HF-style: softmax over all experts → top-k → renormalize
    (matches ``Qwen3_5MoeTopKRouter``).
  * A shared expert path is added: ``sigmoid(gate(x)) * shared_mlp(x)``.

With the default capacity_factor of 1.0 and capacity clamped to
``num_tokens``, no tokens are dropped — required for parity with HF.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Type

import flax.linen as nn
import jax
import jax.numpy as jnp

from theseus.config import configure, field
from theseus.model.axes import Axes
from theseus.model.layers.mlp import QwenMLP
from theseus.model.moe.base import MoE


class _SharedMoEExpertMLP(QwenMLP):
    """QwenMLP variant whose intermediate axis is NOT tagged N_EMBD_FF.

    When this MLP is vmapped across experts and the leading axis is sharded as
    N_EXPERT, we want the intermediate dim replicated (otherwise both axes
    would target the same mesh dim and conflict). Setting the second logical
    axis to ``None`` keeps the intermediate dim out of any global sharding
    rule.
    """

    def setup(self) -> None:
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)
        self.gate = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, None),
            init_std,
        )
        self.up = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, None),
            init_std,
        )
        self.down = self._make_dense(
            self.n_embd,
            (None, Axes.N_EMBD.value),
            proj_std,
        )


class SharedGatedMoE(MoE):
    num_experts: int = field("architecture/num_experts", default=8)
    k: int = field("architecture/num_experts_per_tok", default=2)
    moe_intermediate_size: int = field("architecture/moe_intermediate_size", default=-1)
    shared_expert_intermediate_size: int = field(
        "architecture/shared_expert_intermediate_size", default=-1
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [QwenMLP]

    def _expert_cls(self) -> Type[nn.Module]:
        return _SharedMoEExpertMLP

    def _expert_kwargs(self) -> Dict[str, Any]:
        return {"intermediate_size": self.moe_intermediate_size}

    def _router_partitioning(self) -> Optional[Tuple[Optional[str], Optional[str]]]:
        return (Axes.N_EMBD.value, Axes.N_EXPERT.value)

    def _router_kernel_init(self) -> Any:
        return nn.initializers.zeros

    def setup(self) -> None:
        super().setup()
        self.shared_expert = configure(
            QwenMLP, intermediate_size=self.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Dense(
            1,
            use_bias=False,
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.zeros, (Axes.N_EMBD.value, None)
            ),
        )

    def _select_experts(self, router_logits: jax.Array) -> Tuple[jax.Array, jax.Array]:
        k = min(self.k, self.num_experts)
        probs = nn.softmax(router_logits, axis=-1).astype(jnp.float32)
        top_val, top_idx = jax.lax.top_k(probs, k=k)
        top_val = top_val / jnp.sum(top_val, axis=-1, keepdims=True)
        return top_val, top_idx

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        b, t, h = x.shape
        flat_x = x.reshape(b * t, h)

        shared = self.shared_expert(flat_x, deterministic)
        gate = jax.nn.sigmoid(self.shared_expert_gate(flat_x).astype(jnp.float32))
        shared = gate.astype(shared.dtype) * shared

        routed: jax.Array = super().__call__(x, deterministic)
        out: jax.Array = routed + shared.reshape(b, t, h)
        return out
