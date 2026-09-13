"""Mixture-of-experts feed-forward modules."""

import math
from typing import Any, Dict, List, Optional, Tuple, Type

import flax.linen as nn
import jax
import jax.numpy as jnp

from theseus.base.axis import ShardingPlan
from theseus.config import configure, field
from theseus.model.axes import Axes
from theseus.model.layers.mlp import MLP
from theseus.model.module import Module


class _MoEExpertMLP(MLP):
    """MLP variant whose intermediate axis is NOT tagged N_EMBD_FF.

    When this MLP is vmapped across experts with leading axis sharded as
    N_EXPERT, we want the intermediate dim replicated — otherwise both axes
    target the same mesh dim and conflict.
    """

    def setup(self) -> None:
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)
        self.c_fc = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, None),
            init_std,
        )
        self.c_proj = self._make_dense(
            self.n_embd,
            (None, Axes.N_EMBD.value),
            proj_std,
        )


class MoE(Module):
    """Base MoE feed-forward layer with fixed-capacity expert packing."""

    num_experts: int = field("architecture/moe/experts", default=4)
    k: int = field("architecture/moe/experts_per_embd", default=1)
    capacity_factor: float = field("architecture/moe/capacity_factor", default=1.0)
    capacity_round_to: int = field("architecture/moe/capacity_round_to", default=128)
    instrument_moe: bool = field("architecture/instrumentation/moe", default=False)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [_MoEExpertMLP]

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    # -- hooks for subclasses ---------------------------------------------

    def _expert_cls(self) -> Type[nn.Module]:
        """Expert module class. Subclasses can swap in e.g. a QwenMLP variant."""
        return _MoEExpertMLP

    def _expert_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs passed to `configure(ExpertMLP, ...)`."""
        return {}

    def _router_partitioning(self) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """Partition spec for the router kernel. ``None`` => unpartitioned."""
        return None

    def _router_kernel_init(self) -> Any:
        """Base initializer for the router kernel (before partitioning wrap)."""
        return nn.initializers.normal(stddev=0.02)

    # ---------------------------------------------------------------------

    def setup(self) -> None:
        self._validate_config()

        ExpertMLP = nn.vmap(
            self._expert_cls(),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(0, None),
            out_axes=0,
            axis_size=self.num_experts,
            metadata_params={"partition_name": Axes.N_EXPERT.value},
        )
        self.experts = configure(ExpertMLP, **self._expert_kwargs())

        router_part = self._router_partitioning()
        base_init = self._router_kernel_init()
        if router_part is None:
            router_init = base_init
        else:
            router_init = nn.with_logical_partitioning(base_init, router_part)
        self.router = nn.Dense(
            self.num_experts,
            use_bias=False,
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
            kernel_init=router_init,
        )

    def _validate_config(self) -> None:
        if self.k < 1:
            raise ValueError("architecture/moe/experts_per_embd must be >= 1")
        if self.num_experts < 1:
            raise ValueError("architecture/moe/experts must be >= 1")
        if self.capacity_factor <= 0:
            raise ValueError("architecture/moe/capacity_factor must be > 0")
        if self.capacity_round_to < 1:
            raise ValueError("architecture/moe/capacity_round_to must be >= 1")

    def _select_experts(self, router_logits: jax.Array) -> Tuple[jax.Array, jax.Array]:
        k = min(self.k, self.num_experts)
        _, expert_idx = jax.lax.top_k(router_logits, k=k)
        selected_logits = jnp.take_along_axis(router_logits, expert_idx, axis=-1)
        weights = nn.softmax(selected_logits, axis=-1)
        return weights, expert_idx

    def _pack_assignments(
        self,
        flat_x: jax.Array,
        weights: jax.Array,
        expert_idx: jax.Array,
        capacity: int,
    ) -> Tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        num_tokens, hidden_size = flat_x.shape

        flat_expert_idx = expert_idx.reshape(-1).astype(jnp.int32)
        flat_weights = weights.reshape(-1)
        token_idx = jnp.broadcast_to(
            jnp.arange(num_tokens, dtype=jnp.int32)[:, None],
            expert_idx.shape,
        ).reshape(-1)

        # Group assignments by expert so we can build a single packed expert
        # buffer and run one vmapped expert stack over it.
        order = jnp.argsort(flat_expert_idx, stable=True)
        sorted_expert_idx = flat_expert_idx[order]
        sorted_token_idx = token_idx[order]
        sorted_weights = flat_weights[order]
        sorted_x = flat_x[sorted_token_idx]

        counts = jnp.bincount(sorted_expert_idx, length=self.num_experts)
        expert_offsets = jnp.cumsum(counts, dtype=jnp.int32) - counts
        slot_idx = (
            jnp.arange(sorted_expert_idx.shape[0], dtype=jnp.int32)
            - expert_offsets[sorted_expert_idx]
        )
        keep = slot_idx < capacity
        packed_expert_idx = sorted_expert_idx
        packed_slot_idx = jnp.minimum(slot_idx, capacity - 1)
        packed_x = jnp.where(keep[:, None], sorted_x, 0)

        expert_inputs = jnp.zeros(
            (self.num_experts, capacity, hidden_size),
            dtype=flat_x.dtype,
        )
        expert_inputs = expert_inputs.at[packed_expert_idx, packed_slot_idx].add(
            packed_x
        )

        clipped_counts = jnp.minimum(counts, capacity)
        valid_slots = (
            jnp.arange(capacity, dtype=jnp.int32)[None, :] < clipped_counts[:, None]
        )[..., None]

        return (
            expert_inputs,
            valid_slots,
            packed_expert_idx,
            packed_slot_idx,
            sorted_token_idx,
            sorted_weights,
            keep,
        )

    def _combine_expert_outputs(
        self,
        expert_outputs: jax.Array,
        packed_expert_idx: jax.Array,
        packed_slot_idx: jax.Array,
        sorted_token_idx: jax.Array,
        sorted_weights: jax.Array,
        keep: jax.Array,
        num_tokens: int,
    ) -> jax.Array:
        packed_outputs = expert_outputs[packed_expert_idx, packed_slot_idx]
        weighted_outputs = packed_outputs * sorted_weights[:, None].astype(
            packed_outputs.dtype
        )
        weighted_outputs = jnp.where(keep[:, None], weighted_outputs, 0)
        combined = jnp.zeros(
            (num_tokens, expert_outputs.shape[-1]),
            dtype=weighted_outputs.dtype,
        )
        return combined.at[sorted_token_idx].add(weighted_outputs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        """Apply top-k expert routing to ``x`` of shape ``[B, T, H]``."""

        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        flat_x = x.reshape(num_tokens, hidden_size)

        #### Routing ####

        router_logits = self.router(flat_x).astype(jnp.float32)
        weights, expert_idx = self._select_experts(router_logits)

        #### Expert capacity and packing ####

        assignments = num_tokens * min(self.k, self.num_experts)
        capacity = math.ceil(self.capacity_factor * assignments / self.num_experts)
        round_to = self.capacity_round_to
        capacity = round_to * math.ceil(capacity / round_to)
        capacity = max(1, min(capacity, num_tokens))

        (
            expert_inputs,
            valid_slots,
            packed_expert_idx,
            packed_slot_idx,
            sorted_token_idx,
            sorted_weights,
            keep,
        ) = self._pack_assignments(
            flat_x,
            weights,
            expert_idx,
            capacity=capacity,
        )

        #### MoE diagnostics ####

        if self.instrument_moe:
            assignment_count = expert_idx.size
            counts = jnp.bincount(
                expert_idx.reshape(-1), length=self.num_experts
            ).astype(jnp.float32)
            for expert in range(self.num_experts):
                self.sow(
                    "scalars",
                    f"moe/expert_{expert}_usage",
                    counts[expert] / assignment_count,
                )
            self.sow(
                "scalars",
                "moe/dropped_assignment_fraction",
                1.0 - jnp.mean(keep.astype(jnp.float32)),
            )

        #### Expert execution and combination ####

        expert_outputs = self.experts(expert_inputs, deterministic)
        expert_outputs = jnp.where(valid_slots, expert_outputs, 0)
        combined = self._combine_expert_outputs(
            expert_outputs,
            packed_expert_idx,
            packed_slot_idx,
            sorted_token_idx,
            sorted_weights,
            keep,
            num_tokens,
        )
        return combined.reshape(batch_size, seq_len, hidden_size)
