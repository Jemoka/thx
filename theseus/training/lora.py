"""LoRA (Low-Rank Adaptation) Trainer.

Two-phase trainer: phase 1 trains full parameters on pre-LoRA datasets,
then freezes and injects low-rank adapters, phase 2 trains only LoRA
parameters on post-LoRA datasets.

Since this extends RestoreableJob, users can restore from a checkpoint
and set pre_lora_datasets/tokens to empty to skip straight to LoRA
fine-tuning from a pretrained checkpoint.
"""

from dataclasses import dataclass
from functools import cached_property
from theseus.store import ValueRow
from typing import cast as type_cast
from typing import Any, Dict, List, Optional, Tuple, Type, Generic, TypeVar


import jax
import jax.numpy as jnp
from flax.training import train_state

from loguru import logger

from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Low-rank adaptation parameters."""

    rank: int = field("optimization/lora/rank", default=16)
    alpha: float = field("optimization/lora/alpha", default=16.0)
    target_modules: List[str] = field(
        "optimization/lora/target_modules",
        default_factory=lambda: ["kernel"],
    )


@dataclass
class LoRATrainerConfig(BaseTrainerConfig):
    """Config for two-phase LoRA trainer.

    Pre-LoRA phase trains full parameters, then LoRA adapters are injected
    and only they are trained in the post-LoRA phase.

    LoRA hyperparameters (rank, alpha, target_modules) are read from the
    config namespace via ``configure(LoRAConfig)`` at init time — they
    live under ``optimization/lora/`` in the YAML.
    """

    pre_lora_tokens: List[int] = field(
        "training/pre_lora_tokens",
        default_factory=lambda: [1_000_000_000],
    )

    post_lora_tokens: List[int] = field(
        "training/post_lora_tokens",
        default_factory=lambda: [100_000_000],
    )


# ---------------------------------------------------------------------------
# LoRA parameter utilities
# ---------------------------------------------------------------------------


def param_filter(
    params: PyTree[jax.Array],
    target_modules: List[str],
) -> PyTree[bool]:
    """Create a boolean mask over the param tree.

    Returns True for leaves whose path contains any of ``target_modules``.
    This is the filter that determines which parameters get LoRA adapters.
    """

    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    mask_flat: list[bool] = []
    for keypath, _ in flat:
        path_str = "/".join(str(k) for k in keypath)
        mask_flat.append(any(target in path_str for target in target_modules))
    return treedef.unflatten(mask_flat)  # type: ignore[no-any-return, attr-defined]


def inject_lora_params(
    params: PyTree[jax.Array],
    mask: PyTree[bool],
    rank: int,
    key: jax.Array,
) -> Tuple[PyTree[Optional[jax.Array]], PyTree[Optional[jax.Array]]]:
    """Create LoRA A/B matrices for each targeted parameter.

    For a parameter of shape (in_features, out_features):
    - A: (in_features, rank) — initialized from normal(0, 1/rank)
    - B: (rank, out_features) — initialized to zeros

    Returns two pytrees (lora_A, lora_B) with same structure as params.
    Non-targeted leaves are zero arrays (same shape as base param) so
    the tree structure is compatible with jax.tree_util operations.
    """
    flat_params, treedef = jax.tree_util.tree_flatten(params)
    flat_mask = jax.tree_util.tree_leaves(mask)

    a_flat: list[Any] = []
    b_flat: list[Any] = []
    key_idx = 0
    for param, is_target in zip(flat_params, flat_mask):
        if is_target and param.ndim == 2:
            in_f, out_f = param.shape
            k1 = jax.random.fold_in(key, key_idx)
            A = jax.random.normal(k1, (in_f, rank), dtype=param.dtype) / rank
            B = jnp.zeros((rank, out_f), dtype=param.dtype)
            a_flat.append(A)
            b_flat.append(B)
        else:
            # Non-targeted: use None sentinel
            a_flat.append(None)
            b_flat.append(None)
        key_idx += 1

    return treedef.unflatten(a_flat), treedef.unflatten(b_flat)  # type: ignore[attr-defined]


def merge_lora_params(
    base_params: PyTree[jax.Array],
    lora_A: PyTree[Any],
    lora_B: PyTree[Any],
    alpha: float,
    rank: int,
) -> PyTree[jax.Array]:
    """Merge LoRA into base: W_eff = W + (alpha/rank) * A @ B.

    No ``stop_gradient`` on ``base_params`` is needed: in the training
    loop ``value_and_grad`` differentiates only w.r.t. its argument
    (the LoRA params dict), so ``base_params`` — accessed from the
    closed-over state — is already treated as a constant by JAX.
    """
    scale = alpha / rank

    def _merge(base: jax.Array, a: Any, b: Any) -> jax.Array:
        if a is None or b is None:
            return base
        delta = scale * (a @ b)
        return base + delta.astype(base.dtype)  # type: ignore[no-any-return]

    return jax.tree_util.tree_map(  # type: ignore[no-any-return]
        _merge,
        base_params,
        lora_A,
        lora_B,
        is_leaf=lambda x: x is None,
    )


def count_lora_params(lora_A: PyTree[Any], lora_B: PyTree[Any]) -> int:
    """Count total trainable LoRA parameters."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(lora_A):
        if leaf is not None and hasattr(leaf, "size"):
            total += leaf.size
    for leaf in jax.tree_util.tree_leaves(lora_B):
        if leaf is not None and hasattr(leaf, "size"):
            total += leaf.size
    return total


# ---------------------------------------------------------------------------
# Custom train state
# ---------------------------------------------------------------------------


class LoRATrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    """Train state for LoRA phase.

    ``params`` holds the trainable LoRA parameters as
    ``{"lora_A": pytree, "lora_B": pytree}``.  The optimizer in
    ``tx`` acts on ``params``, so only the adapters are updated.

    ``base_params`` holds the frozen pretrained weights (bfloat16).
    """

    base_params: PyTree[Any]  # frozen base parameters
    lora_alpha: float
    lora_rank: int

    def merge_params(self, params: Dict[str, Any]) -> PyTree[jax.Array]:
        """Materialize model weights for training or inference from these adapters."""
        return merge_lora_params(
            self.base_params,
            params["lora_A"],
            params["lora_B"],
            self.lora_alpha,
            self.lora_rank,
        )


# ---------------------------------------------------------------------------
# Shared LoRA transition logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

C = TypeVar("C", bound=LoRATrainerConfig)


class LoRATrainer(BaseTrainer[C, M], Generic[C, M]):
    """Two-phase LoRA trainer.

    Phase 1 (pre-LoRA): Full-parameter training on ``pre_lora_datasets``.
    Phase 2 (post-LoRA): Freeze base, inject LoRA adapters, train only
    adapters on ``post_lora_datasets``.

    The transition happens automatically at the token boundary between
    pre-LoRA and post-LoRA phases.
    """

    CONFIG = LoRATrainerConfig  # type: ignore[assignment]

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [LoRAConfig]

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        assert spec.topology is not None
        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas
        self.total_steps = int(
            (sum(self.args.pre_lora_tokens) + sum(self.args.post_lora_tokens))
            / self.args.batch_size
            / self.args.block_size
        )
        return topology

    # ------------------------------------------------------------------
    # State – starts as standard TrainState; switches to LoRATrainState
    # ------------------------------------------------------------------

    @cached_property
    def lora_config(self) -> LoRAConfig:
        return type_cast(LoRAConfig, configure(LoRAConfig))

    def _make_state(self, params: PyTree[jax.Array]) -> train_state.TrainState:
        if not self._in_lora_phase:
            return super()._make_state(params)
        cfg = self.lora_config
        mask = param_filter(params, cfg.target_modules)
        lora_A, lora_B = inject_lora_params(params, mask, cfg.rank, self.init_key)
        return LoRATrainState.create(  # type: ignore[no-any-return, no-untyped-call]
            apply_fn=self.model.apply,
            params={"lora_A": lora_A, "lora_B": lora_B},
            base_params=jax.tree.map(lambda value: value.astype(jnp.bfloat16), params),
            tx=self.optimizer,
            lora_alpha=cfg.alpha,
            lora_rank=cfg.rank,
        )

    def _transition_to_lora(self) -> None:
        self._in_lora_phase = True
        sharding = jax.tree.map(lambda value: value.sharding, self.template)
        state = jax.jit(self._make_state, out_shardings=sharding)(self.state.params)
        # Preserve the canonical training clock across the parameter change.
        self.state = state.replace(step=self.state.step)
        self.state_sharding = sharding
        logger.info("LORA | adapters initialized (rank={})", self.lora_config.rank)

    def state_restore(self, state: ValueRow) -> None:
        self._in_lora_phase = bool(state.get("lora_phase", False))
        super().state_restore(state)

    def checkpoint(self) -> None:
        self.save(
            self.state,
            {
                "lora_phase": self._in_lora_phase,
            },
        )
        self.log({"checkpoint": 1})

    def _init_data(self, spec: ExecutionSpec) -> None:
        self._pre_lora_total = sum(self.args.pre_lora_tokens)
        self._in_lora_phase = self._pre_lora_total == 0
        self.lora_config
        super()._init_data(spec)

    def apply(self, state: PyTree[Any], metadata: Dict[str, Any]) -> None:
        super().apply(state, metadata)
        tokens = int(self.state.step) * self.args.batch_size * self.args.block_size
        if not self._in_lora_phase and tokens >= self._pre_lora_total:
            self._transition_to_lora()

    def tick(self) -> None:
        tokens = int(self.state.step) * self.args.batch_size * self.args.block_size
        if not self._in_lora_phase and tokens >= self._pre_lora_total:
            self._transition_to_lora()
            self.checkpoint()
        super().tick()
