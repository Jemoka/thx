"""
KL Divergence Trainer

Two-stage trainer: stage 1 is standard pretraining, stage 2 enforces a
customizable KL penalty against the stage-1 reference policy.

Stage switching is controlled by token budgets.
"""

from functools import cached_property
from theseus.training.schedules import WSDS
from dataclasses import dataclass
from typing import cast as type_cast
from typing import Any, Dict, Optional, List, Type, Generic, TypeVar

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jax_random
from flax.training import train_state
from jax.experimental import multihost_utils

from loguru import logger

from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.utils import scalar_metadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KLConfig:
    """KL divergence penalty configuration."""

    beta: float = field("optimization/kl/beta", default=0.1)


@dataclass
class KLDivergenceTrainerConfig(BaseTrainerConfig):
    """Config for two-stage KL-divergence trainer.

    ``total_tokens`` is a two-element list: [stage1_tokens, stage2_tokens].
    Both stages use the trainer's static ``DATASET`` declaration.
    """

    total_tokens: List[int] = field(
        "training/tokens",
        default_factory=lambda: [1_000_000_000, 100_000_000],
    )  # type: ignore


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------


class KLDivergenceTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    """Train state carrying a frozen reference-policy snapshot and KL weight."""

    base: PyTree[Any]  # reference policy params (frozen)
    beta: jax.Array  # KL penalty weight (0 = disabled)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

C = TypeVar("C", bound=KLDivergenceTrainerConfig)


class KLDivergenceTrainer(BaseTrainer[C, M], Generic[C, M]):
    """Two-stage trainer with KL-divergence penalty.

    * **Stage 1** – standard language-model pretraining (cross-entropy only).
    * **Stage 2** – pretraining loss *plus* ``beta * KL(policy || reference)``
      where the reference policy is a frozen snapshot taken at the stage
      boundary.

    The KL penalty is approximated as the difference in per-token NLL
    between the current model and the reference model on the same batch:
    ``kl_penalty = policy_loss - sg(reference_loss)``.
    """

    CONFIG = KLDivergenceTrainerConfig  # type: ignore[assignment]

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [KLConfig]

    SCHEDULE = WSDS

    # ------------------------------------------------------------------
    # Topology – total tokens are the *sum* of all stages
    # ------------------------------------------------------------------

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        assert spec.topology is not None, (
            "Topology must be provided to perform training"
        )
        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas
        self.total_steps = int(
            sum(self.args.total_tokens) / self.args.batch_size / self.args.block_size
        )
        return topology

    # ------------------------------------------------------------------
    # State – includes reference-policy slot
    # ------------------------------------------------------------------

    def _make_state(self, params: PyTree[jax.Array]) -> KLDivergenceTrainState:
        return KLDivergenceTrainState.create(  # type: ignore[no-any-return, no-untyped-call]
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            base=jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params),
            beta=jnp.asarray(0.0),
        )

    @cached_property
    def kl_config(self) -> KLConfig:
        return type_cast(KLConfig, configure(KLConfig))

    def _init_data(self, spec: ExecutionSpec) -> None:
        super()._init_data(spec)
        if len(self.args.total_tokens) != 2:
            raise ValueError("KL training requires exactly two token budgets")
        self._segment_ends = np.cumsum(self.args.total_tokens).tolist()
        self._current_stage = 0

    def apply(self, state: PyTree[Any], metadata: Dict[str, Any]) -> None:
        super().apply(state, metadata)
        self._current_stage = self._stage_for_token(self._current_token_position())

    def _current_token_position(self) -> int:
        return int(self.state.step) * self.args.batch_size * self.args.block_size

    def _stage_for_token(self, ntok: int) -> int:
        """Return the stage index for a given token position."""
        for i, end in enumerate(self._segment_ends):
            if ntok < end:
                return i
        return len(self._segment_ends) - 1

    def _snapshot_reference(self) -> None:
        """Snapshot current params into state.base as the reference policy.

        The cast operates element-wise on each host's local shards –
        no cross-host gather is performed and sharding is preserved.
        Barriers ensure every host completes the snapshot before any
        host proceeds to use the new reference.
        """
        multihost_utils.sync_global_devices("kl_snapshot:start")
        # .astype on sharded arrays preserves sharding; each host only
        # touches its local shards, so no full-param materialisation.
        new_base = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), self.state.params
        )
        self.state = self.state.replace(base=new_base)
        multihost_utils.sync_global_devices("kl_snapshot:end")
        if self.main_process():
            logger.info("KL | reference policy snapshot taken")

    def _on_stage_boundary(self, old_stage: int, new_stage: int) -> None:
        """Called when transitioning between stages.

        By default, snapshots the reference policy when entering stage 1
        (i.e. the second stage, index 1).  Subclasses may override for
        more complex behaviour.
        """
        if self.inference is not None:
            self.log(self.inference.evaluate())
        self.log(
            {
                "stage/index": new_stage,
                "stage/switch_at_tokens": self._current_token_position(),
            }
        )

        # Snapshot reference policy and activate KL penalty when entering
        # stage 1 (the KL stage).
        if new_stage >= 1:
            self._snapshot_reference()
            state = type_cast(KLDivergenceTrainState, self.state)
            self.state = self.state.replace(
                beta=jax.device_put(
                    jnp.asarray(self.kl_config.beta, dtype=state.beta.dtype),
                    state.beta.sharding,
                )
            )
            if self.main_process():
                logger.info("KL | beta set to {}", self.kl_config.beta)

        self.checkpoint()

    def tick(self) -> None:
        new_stage = self._stage_for_token(self._current_token_position())
        if new_stage != self._current_stage:
            self._on_stage_boundary(self._current_stage, new_stage)
            self._current_stage = new_stage
        super().tick()

    # ------------------------------------------------------------------
    # Forward – standard CE + KL penalty (stage >= 1 only)
    # ------------------------------------------------------------------

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        kl_state = type_cast(KLDivergenceTrainState, state)
        batch_dict = type_cast(Dict[str, jax.Array], batch)

        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)
        rngs = {"dropout": dropout_key} if dropout_key is not None else {}

        # Policy forward pass
        mutable = ["scalars"]
        if intermediates:
            mutable.extend(("intermediates", "plots"))

        (logits, policy_loss), mutated = kl_state.apply_fn(
            {"params": params},
            x,
            y,
            padding_mask=padding_mask,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
        )
        meta: Dict[str, Any] = scalar_metadata(mutated.get("scalars", {}))
        if intermediates:
            meta.update(
                {
                    "intermediates": mutated.get("intermediates", {}),
                    "plots": mutated.get("plots", {}),
                }
            )

        # KL penalty: beta * (policy_loss - sg(reference_loss))
        # The reference forward always executes (beta is a traced value),
        # but in stage 0 beta=0 so the penalty is zeroed out numerically.
        beta = kl_state.beta

        ref_loss = jax.lax.stop_gradient(
            kl_state.apply_fn(
                {"params": kl_state.base},
                x,
                y,
                padding_mask=padding_mask,
                deterministic=True,
            )[1]
        )

        kl_penalty = policy_loss - ref_loss
        total_loss = policy_loss + beta * kl_penalty

        meta.update(
            {
                "kl/policy_loss": jax.lax.stop_gradient(policy_loss),
                "kl/ref_loss": ref_loss,
                "kl/penalty": jax.lax.stop_gradient(kl_penalty),
                "kl/beta": beta,
                "kl/total_loss": jax.lax.stop_gradient(total_loss),
            }
        )

        return logits, total_loss, meta
