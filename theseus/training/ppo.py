"""On-policy PPO trainer.

Custom train state caches a frozen reference policy (snapshotted at init) so
the loss can apply a KL penalty against it. The training step is unchanged
relative to BaseTrainer — the loop calls `self.batch()` per step, which we
override to (a) roll out the current policy via the existing `Evaluator`
machinery for every RL component, (b) build an (N, k) per-rollout per-component
reward matrix where each row's source-component column holds that component's
score and other columns are zero, (c) optionally transform per-component via
`self.reward_postprocess(R)`, (d) gather the per-rollout scalar from each row's
source column and smear it per-token via reward-to-go, and (e) cache
`old_log_probs` for the importance ratio.

Subclasses (e.g. `GRPOTrainer`) override advantage computation by overriding
`_advantages_from_rewards`.
"""

from functools import cached_property
from dataclasses import dataclass
from typing import cast as type_cast
from typing import Any, Dict, Optional, List, Type, Generic, Tuple

import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jax_random
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

from loguru import logger

from theseus.base import PyTree, Axis, ExecutionSpec
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.backbone import BackbonedTrainer
from theseus.training.utils import scalar_metadata
from theseus.model.module import Module
from theseus.evaluation.base import Evaluator, Evaluation


@dataclass
class PPOConfig:
    beta: float = field("optimization/ppo/beta", default=0.04)  # KL penalty
    clip_eps: float = field("optimization/ppo/clip_eps", default=0.2)
    discount: float = field("optimization/ppo/discount", default=1.0)
    sample_temperature: float = field(
        "optimization/ppo/sample_temperature", default=1.0
    )
    sample_top_p: float = field("optimization/ppo/sample_top_p", default=1.0)


class PPOTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    base: PyTree[Any]
    beta: float
    clip_eps: float


class PPOTrainer(BaseTrainer[BaseTrainerConfig, M], Generic[M]):
    """PPO trainer scaffold."""

    CONFIG = BaseTrainerConfig
    _batch_seq: int | None = None
    _batch: PyTree[np.ndarray]
    DATASET = []
    RL_EVALUATION: List[Type[Evaluation]] = []

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [PPOConfig, *Evaluator.config(cls.RL_EVALUATION)]

    def reward_postprocess(self, scores: np.ndarray) -> np.ndarray:
        """Optional per-component reward transformation. Shape-preserving
        (N, k) -> (N, k). Most subclasses do NOT need to override this — the
        default identity is correct whenever each component's raw score is
        already the desired training signal.

        ``scores[n, i]`` is component ``i``'s score on rollout ``n``. Components
        have disjoint datasets, so each rollout is produced and scored by exactly
        one component: only the source column is populated per row, other columns
        are zero (structural padding so the matrix has uniform shape, not a
        signal). Column order matches ``RL_EVALUATION`` order —
        i.e. the order of ``self.rl_inference.evaluations``.

        Override only to scale, clip, or otherwise transform individual channels
        (e.g. ``scores[:, 0] *= 2.0`` to upweight the first component, or to
        clip a noisy reward to [0, 1]). Must preserve the zero-padding for
        non-source columns if you want per-component logging means to remain
        meaningful — see the source-mask trick in `forward()`.

        Called from ``_refill_buffer`` outside the gradient path; subclasses
        can freely read instance state here. After this returns, the per-
        rollout scalar fed to PPO is just ``R[n, source[n]]`` — there is no
        cross-channel aggregation.
        """
        return scores.astype(np.float32)

    # -- state -----------------------------------------------------------------

    def _make_state(self, params: PyTree[jax.Array]) -> PPOTrainState:
        cfg = configure(PPOConfig)
        return PPOTrainState.create(  # type: ignore[no-any-return, no-untyped-call]
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            base=jax.tree.map(lambda x: x.astype(jnp.bfloat16), params),
            beta=cfg.beta,
            clip_eps=cfg.clip_eps,
        )

    @cached_property
    def ppo_config(self) -> PPOConfig:
        return type_cast(PPOConfig, configure(PPOConfig))

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Skip the dataset — PPO generates batches from rollouts."""
        self.strategy = None  # type: ignore[assignment]
        self.train_dl = None  # type: ignore[assignment]
        self.val_dl = None  # type: ignore[assignment]

    def _init_counters_and_eval(self) -> None:
        """Standard counters + an extra Evaluator for the RL components."""
        super()._init_counters_and_eval()
        self._batch_seq = None
        self.rl_inference: Evaluator[M] = Evaluator.from_trainer(self)
        self.rl_inference.evaluations = [
            evaluation() for evaluation in self.RL_EVALUATION
        ]

        # Buffer entries are
        # (x, padding_mask, action_mask, old_log_probs, reward, component_row).
        # `old_log_probs` is captured at rollout time under the rollout-generating
        # policy (= π_θ_old in standard PPO notation), so the importance ratio in
        # the surrogate is exactly π_θ_new / π_θ_old. `component_row` is the
        # (k,) per-rollout reward vector from the (N, k) score matrix (after
        # ``self.reward_postprocess`` is applied) — only the source-component
        # column is populated, others are zero. Kept solely for per-component
        # metric logging; the scalar `reward` is what actually feeds the
        # objective.
        self._rollout_buffer: List[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray,
            ]
        ] = []
        # Component names in matrix-column order (matches RL_EVALUATION
        # `components` order). Captured once at init so JIT shapes are stable.
        self._component_names: List[str] = [
            e.name for e in self.rl_inference.evaluations
        ]
        # Cached JIT for the rollout-time log-prob pass (built lazily on first use).
        self._old_logp_jit: Optional[Any] = None

        if self.main_process():
            logger.info(self.model)
            logger.info(
                "PPO | RL components: {}",
                [e.name for e in self.rl_inference.evaluations],
            )
            logger.debug(
                "PPO | per-step rollouts requested: {}",
                self.per_device_batch_size
                * self.local_replicas
                * self.accumulate_steps,
            )

    # -- rollout collection ----------------------------------------------------

    def _samples_per_prompt(self) -> int:
        """How many rollouts to draw per unique prompt during a refill.

        Default 1 (vanilla PPO). Subclasses that need group-relative semantics
        (GRPO, RLOO) override this; the evaluator then replicates each selected
        prompt index G times so the buffer arrives with G consecutive same-
        prompt rollouts per slot.
        """
        return 1

    def _refill_buffer(self) -> None:
        """Run the RL evaluator across ALL components, build per-rollout rewards
        from each component's own scoring of its own dataset, and stage entries
        as (x, padding_mask, action_mask, old_log_probs, reward, component_row)
        tuples on every host."""
        cfg = self.ppo_config

        # train_step donates self.state, so the rl_inference's snapshot is
        # stale after step 0 — refresh it before every rollout.

        logger.debug(
            "PPO | refilling rollout buffer (components={}, temperature={}, top_p={})",
            self._component_names,
            cfg.sample_temperature,
            cfg.sample_top_p,
        )
        evals_dict, rollouts_per_eval = self.rl_inference.evaluate(
            reduce="none",
            return_intermediates=True,
            temperature=cfg.sample_temperature,
            top_p=cfg.sample_top_p,
            samples_per_prompt=self._samples_per_prompt(),
        )
        logger.debug("PPO | rollouts done; computing rewards")

        if not rollouts_per_eval or not any(rollouts_per_eval):
            raise RuntimeError(
                "PPO | RL evaluator produced no rollouts. Make sure "
                "RL_EVALUATION declares at least one nonempty evaluation."
            )

        # Component column order: list(evals_dict) preserves insertion order =
        # iteration order over self.rl_inference.evaluations = matrix layout.
        component_names = list(evals_dict.keys())
        if component_names != self._component_names:
            raise RuntimeError(
                f"PPO | evaluator component order {component_names} drifted from "
                f"cached {self._component_names}; matrix columns would mismatch."
            )
        k = len(component_names)
        if len(rollouts_per_eval) != k:
            raise RuntimeError(
                f"PPO | rollouts_per_eval has {len(rollouts_per_eval)} groups "
                f"but evals_dict has {k} components; expected one rollout list "
                f"per component."
            )

        # Per-component co-indexing check: each component's score array must
        # have one entry per rollout produced by that component.
        per_eval_counts = [len(r) for r in rollouts_per_eval]
        for i, (name, n_i) in enumerate(zip(component_names, per_eval_counts)):
            score_arr = np.asarray(evals_dict[name], dtype=np.float32)
            if score_arr.shape != (n_i,):
                raise ValueError(
                    f"PPO | component '{name}' has {n_i} rollouts but score "
                    f"shape {score_arr.shape}; expected ({n_i},). Each "
                    f"component scores its own rollouts; the two arrays must "
                    f"be co-indexed."
                )

        # ──────────────────────────────────────────────────────────────────
        # ORDERING CONTRACT — DO NOT SHUFFLE `rollouts`, `R`, or `source`
        # past this point. They are co-indexed, and:
        #   (a) GRPO (via _samples_per_prompt) relies on G consecutive same-
        #       prompt slots: [p0_s0..p0_s(G-1), p1_s0..p1_s(G-1), ...].
        #       Concatenation across components preserves this because each
        #       component's count N_i is a multiple of G (samples_per_prompt
        #       is applied per-component inside RolloutEvaluation).
        #   (b) `source[n]` identifies which component scored rollout n, and
        #       only column source[n] of R is populated for row n. Any reorder
        #       silently mismatches rollouts to scores AND breaks GRPO groups.
        # If you need stochastic order, do it BEFORE _select_indices in the
        # evaluator or AFTER _smear_rewards has produced per-token rewards.
        # ──────────────────────────────────────────────────────────────────
        rollouts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [
            r for component_rollouts in rollouts_per_eval for r in component_rollouts
        ]
        N = len(rollouts)

        # Build (N, k) score matrix + source index. By construction, only one
        # entry per row is populated (R[n, source[n]]); other entries are zero.
        # The other k-1 columns are structural padding so reward() has uniform
        # shape — they are NOT a "no-reward" signal and should not be summed.
        R = np.zeros((N, k), dtype=np.float32)
        source = np.zeros(N, dtype=np.int32)
        offset = 0
        for i, name in enumerate(component_names):
            n_i = per_eval_counts[i]
            R[offset : offset + n_i, i] = np.asarray(evals_dict[name], dtype=np.float32)
            source[offset : offset + n_i] = i
            offset += n_i

        # Per-component reward transformation hook (default identity). Subclasses
        # may scale/clip individual columns; shape must be preserved.
        R = np.asarray(self.reward_postprocess(R), dtype=np.float32)
        if R.shape != (N, k):
            raise ValueError(
                f"PPO | self.reward_postprocess returned shape {R.shape}, "
                f"expected ({N}, {k}). reward_postprocess() is shape-preserving "
                f"— it transforms, not aggregates."
            )

        # Per-rollout scalar = source component's (post-transform) score.
        # Source-gather, not sum: the other k-1 entries in row n are padding,
        # not contributions to be aggregated.
        rewards = R[np.arange(N), source]

        # The rollout source should have uniform sequence length T because each
        # eval pads to one ``total_tokens`` and every component shares the
        # trainer's block_size.
        ts = {r[0].shape[-1] for r in rollouts}
        if len(ts) > 1:
            raise NotImplementedError(
                f"PPO | rollouts produced mixed sequence lengths {sorted(ts)} "
                f"across all components; expected one padded T."
            )

        # Cache log π_θ_old(y_t | x_<=t) NOW, under the rollout-generating
        # policy. Bound to π_θ_old by construction — the PPO contract.
        x_arr = np.stack([r[0] for r in rollouts]).astype(np.int32)
        am_arr = np.stack([r[1] for r in rollouts]).astype(bool)
        pm_arr = np.stack([r[2] for r in rollouts]).astype(np.int32)
        y_arr = np.roll(x_arr, -1, axis=-1)
        y_arr[:, -1] = 0
        y_safe = np.where(am_arr, y_arr, 0).astype(np.int32)
        logger.debug(
            "PPO | computing old log_probs (B={} T={})", x_arr.shape[0], x_arr.shape[-1]
        )
        old_log_probs = self._rollout_log_probs(x_arr, y_safe, pm_arr)
        old_log_probs = np.where(am_arr, old_log_probs, 0.0).astype(np.float32)

        # Multi-host: rollouts/R/log_probs are all replicated on every host (the
        # eval allgather'd). The SPMD train_step expects each host to provide its
        # UNIQUE shard, which _to_global will stitch back into the global batch.
        # Split here so each host's buffer holds only its slice.
        # WARNING: np.array_split returns CONTIGUOUS index ranges per host.
        # Do not switch this to a strided/round-robin/shuffled split — GRPO's
        # group_size G must divide each host's slice cleanly so groups are never
        # cut across hosts. The B%G assertion in GRPO._smear_rewards is the
        # backstop, but a strided split would *pass* the assertion while
        # shattering same-prompt grouping. Keep it contiguous.
        n_hosts, host_idx = jax.process_count(), jax.process_index()
        my_indices = np.array_split(np.arange(N), n_hosts)[host_idx]

        new_entries: List[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray,
            ]
        ] = [
            (
                rollouts[i][0],
                rollouts[i][2].astype(bool),  # padding_mask
                rollouts[i][1].astype(bool),  # action_mask
                old_log_probs[i],
                float(rewards[i]),
                R[i],  # (k,) per-rollout component vector for logging
            )
            for i in my_indices
        ]
        # `new_entries` is iterated in `my_indices` order (contiguous range);
        # do not insert a shuffle on this list — see the ORDERING CONTRACT.
        self._rollout_buffer.extend(new_entries)
        logger.debug(
            "PPO | refill: B_global={} (sum N_i={}) → host {}/{} got {} (T={}, action_tokens/sample={:.1f})",
            N,
            per_eval_counts,
            host_idx,
            n_hosts,
            len(new_entries),
            x_arr.shape[-1],
            float(am_arr.sum() / max(am_arr.shape[0], 1)),
        )

    def _smear_rewards(
        self, rewards: np.ndarray, action_mask: np.ndarray, discount: float
    ) -> np.ndarray:
        """Distribute terminal sequence rewards across action positions via
        reward-to-go: R_t = γ^(T_a - 1 - t_a) * r where t_a indexes only the
        action positions. With γ=1, every action token gets the sequence reward.
        Non-action positions get 0.
        """
        B, T = action_mask.shape
        out = np.zeros((B, T), dtype=np.float32)
        if discount == 1.0:
            # np.repeat builds [r_0]*n_0 + [r_1]*n_1 + ...; boolean indexing on
            # action_mask iterates row-major (same order), so each row's True
            # slots get filled with that row's reward.
            out[action_mask] = np.repeat(rewards, action_mask.sum(axis=-1))
            return out
        # General case: enumerate action positions per row.
        for b in range(B):
            idxs = np.flatnonzero(action_mask[b])
            if idxs.size == 0:
                continue
            n = idxs.size
            # Largest exponent at the first action position; 0 at the last.
            powers = np.power(discount, np.arange(n - 1, -1, -1, dtype=np.float32))
            out[b, idxs] = rewards[b] * powers
        return out

    # -- rollout-time log-probs ------------------------------------------------

    @staticmethod
    def log_prob_step(
        state: train_state.TrainState,
        batch: PyTree[jax.Array],  # dict with x/y/padding_mask, each (S, B, T)
    ) -> jax.Array:
        """Scan log π(y_t | x_<=t) over S micro-batches; mirrors val_step shape."""

        def reduce(_: Any, batch_item: Any) -> Any:
            params: PyTree[jax.Array] = type_cast(PyTree[jax.Array], state.params)
            logits, _, _ = BaseTrainer.forward(
                state, params, batch_item, deterministic=True
            )
            lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            chosen = jnp.take_along_axis(
                lp, batch_item["y"][..., None], axis=-1
            ).squeeze(-1)
            return None, chosen

        _, log_probs = jax.lax.scan(reduce, None, batch)  # (S, B, T)
        return type_cast(jax.Array, log_probs)

    def _rollout_log_probs(
        self,
        x: np.ndarray,
        y: np.ndarray,
        padding_mask: np.ndarray,
    ) -> np.ndarray:
        """Cache log π_θ_old(y_t | x_<=t) for the freshly-drawn rollout batch.

        Inputs are ``(B_global, T)`` numpy, *replicated* on every host (the
        eval ``process_allgather``-s rollouts so every host stores the full
        set). To avoid every host redoing the same forward, we split the
        replicated batch into unique per-host shards, then defer to the same
        ``_reshape_batch`` / ``_to_global`` plumbing the train loop uses.
        After the scan we ``process_allgather`` so every host's buffer ends
        up with matching ``(B_global, T)`` log-probs.
        """
        from jax.experimental import multihost_utils as _mh

        # 1. Split replicated → host-local, then use the parent's standard
        #    (S, B, T) reshape + global-array conversion.
        n_hosts, host_idx = jax.process_count(), jax.process_index()
        local_batch: PyTree[np.ndarray] = type_cast(
            PyTree[np.ndarray],
            {
                "x": np.array_split(x, n_hosts, axis=0)[host_idx],
                "y": np.array_split(y, n_hosts, axis=0)[host_idx],
                "padding_mask": np.array_split(padding_mask, n_hosts, axis=0)[host_idx],
            },
        )
        batch = self._to_global(self._reshape_batch(local_batch))

        # 2. JIT once with the same data sharding as the train step.
        if self._old_logp_jit is None:
            data_shard = NamedSharding(self.mesh, P(None, Axis.BATCH, None))  # type: ignore[no-untyped-call]
            self._old_logp_jit = jax.jit(
                self.log_prob_step,
                in_shardings=(self.state_sharding, data_shard),
                out_shardings=data_shard,
            )

        with (
            jax.set_mesh(self.mesh),
            nn.logical_axis_rules(self.model.sharding._tp),
        ):
            log_probs_g = self._old_logp_jit(self.state, batch)  # (S, B, T) sharded

        # 3. Local shard → numpy, allgather across hosts.
        log_probs_local = _mh.global_array_to_host_local_array(
            log_probs_g,
            self.mesh,
            P(None, Axis.BATCH, None),  # type: ignore[no-untyped-call]
        )
        if n_hosts > 1:
            # tiled=True: concatenate host-local chunks along the leading
            # microbatch axis; flattening happens after gather on host.
            gathered = _mh.process_allgather(log_probs_local, tiled=True)
        else:
            gathered = log_probs_local
        gathered_np = np.asarray(gathered).reshape(-1, gathered.shape[-1])
        logger.debug(
            "PPO | rollout-time log_probs ready (host {}/{}, B_global={} T={})",
            host_idx,
            n_hosts,
            gathered_np.shape[0],
            gathered_np.shape[-1],
        )
        return gathered_np

    # -- batch -----------------------------------------------------------------

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        """Generate one PPO training batch by reusing the Evaluator dynamics.

        Returns a numpy dict shaped to feed straight into the parent train()'s
        ``_to_global(_reshape_batch(...))``.
        """
        if self._batch_seq == self.node.seq:
            return self._batch
        bsz = self.per_device_batch_size * self.local_replicas * self.accumulate_steps
        while len(self._rollout_buffer) < bsz:
            logger.debug(
                "PPO | buffer have={} need={}, refilling",
                len(self._rollout_buffer),
                bsz,
            )
            self._refill_buffer()

        # ──────────────────────────────────────────────────────────────────
        # ORDERING CONTRACT — DO NOT SHUFFLE `entries`.
        # The buffer is FIFO and stores rollouts in the order produced by
        # _refill_buffer, which (for GRPO and any other group-relative
        # subclass) arrives as G consecutive same-prompt slots. Every
        # downstream array in this batch (x, padding_mask, action_mask,
        # old_log_probs, rewards, component_rewards) is co-indexed off
        # `entries` and fed into _smear_rewards in the same order. Shuffling
        # here turns GRPO's z-score into a no-op — silently. If you need
        # randomization, do it BEFORE rollout collection (prompt sampling)
        # or AFTER _smear_rewards has produced per-token rewards.
        # ──────────────────────────────────────────────────────────────────
        entries = self._rollout_buffer[:bsz]
        self._rollout_buffer = self._rollout_buffer[bsz:]

        x = np.stack([e[0] for e in entries]).astype(np.int32)  # (B, T)
        padding_mask = np.stack([e[1] for e in entries]).astype(bool)
        action_mask = np.stack([e[2] for e in entries]).astype(bool)
        old_log_probs = np.stack([e[3] for e in entries]).astype(np.float32)
        rewards = np.array([e[4] for e in entries], dtype=np.float32)  # (B,)
        # Per-rollout (k,) reward rows from `_refill_buffer` stacked into (B, k).
        # Column order matches `self._component_names` (fixed at init time, so
        # JIT shapes are stable). Each row has exactly one populated entry —
        # the rollout's source component — and zeros elsewhere.
        component_names = self._component_names
        component_matrix = (
            np.stack([e[5] for e in entries], axis=0).astype(np.float32)
            if entries
            else np.zeros((0, len(component_names)), dtype=np.float32)
        )
        component_rewards_per_rollout: Dict[str, np.ndarray] = {
            name: component_matrix[:, i] for i, name in enumerate(component_names)
        }

        # y = next-token target (shift x left by 1); zero outside action_mask
        # since the loss only reads y at action positions anyway.
        y = np.roll(x, -1, axis=-1)
        y[:, -1] = 0
        y_safe = np.where(action_mask, y, 0).astype(np.int32)

        # Per-token reward via reward-to-go with discount γ. (Subclasses like
        # GRPO normalize within group inside _smear_rewards, so this array is
        # really "advantage per token" by the time it leaves.)
        per_token_rewards = self._smear_rewards(
            rewards, action_mask, self.ppo_config.discount
        )
        # Carry raw component scores through the same (S, B, *) sharding plumbing
        # as logging-only scalar channels. They intentionally do not use
        # _smear_rewards, because GRPO normalization and action-token weighting
        # are objective concerns, not component metric concerns.
        component_rewards: Dict[str, np.ndarray] = {
            name: arr[:, None].astype(np.float32)
            for name, arr in component_rewards_per_rollout.items()
        }

        # Stash raw per-rollout reward stats so we can log "is the policy
        # actually learning" without GRPO's z-score eating the signal.
        self._last_raw_reward_mean = float(rewards.mean())
        self._last_raw_reward_max = float(rewards.max())
        if self.main_process():
            logger.debug(
                "PPO | raw_reward mean={:.3f} max={:.3f} (B={})",
                self._last_raw_reward_mean,
                self._last_raw_reward_max,
                rewards.shape[0],
            )
            for name, arr in component_rewards_per_rollout.items():
                # Mean over rollouts where this component is the source — i.e.
                # rows whose populated column is this one. Non-source rows are
                # zero by construction; using `arr.mean()` would dilute the
                # signal by 1/k. Detect source rows via "non-zero entry"; this
                # is a safe heuristic because the default reward() preserves
                # the zero padding. Subclasses that add per-channel biases
                # should override this debug log.
                source_count = int((arr != 0.0).sum())
                source_mean = float(arr.sum() / max(source_count, 1))
                source_max = float(arr.max()) if source_count > 0 else 0.0
                logger.debug(
                    "PPO | component[{}] source_mean={:.3f} source_max={:.3f} (n={})",
                    name,
                    source_mean,
                    source_max,
                    source_count,
                )
            logger.debug(
                "PPO | per-token reward stats: mean={:.4f} std={:.4f} action_tokens/sample={:.1f}",
                float(per_token_rewards.mean()),
                float(per_token_rewards.std()),
                float(action_mask.sum() / max(action_mask.shape[0], 1)),
            )

        # Cast: the dict literal's inferred value type widens to include
        # ``Dict[str, np.ndarray]`` (component_rewards), which mypy can't
        # auto-unify with ``PyTree[np.ndarray]`` even though it's structurally
        # valid (PyTree[T] = T | list[...] | tuple[...] | dict[str, PyTree[T]]).
        self._batch = type_cast(
            PyTree[np.ndarray],
            {
                "x": x,
                "y": y_safe,
                "padding_mask": padding_mask.astype(np.int32),
                "action_mask": action_mask.astype(np.int32),
                "old_log_probs": old_log_probs,
                "per_token_rewards": per_token_rewards,
                "component_rewards": component_rewards,
            },
        )

        self._batch_seq = self.node.seq
        return self._batch

    # -- forward / loss --------------------------------------------------------

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        """PPO clipped surrogate loss + KL penalty against the frozen reference."""
        cstate = type_cast(PPOTrainState, state)
        batch_dict = type_cast(Dict[str, jax.Array], batch)

        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"].astype(jnp.bool_)
        action_mask = batch_dict["action_mask"].astype(jnp.bool_)
        old_log_probs = batch_dict["old_log_probs"]
        per_token_rewards = batch_dict["per_token_rewards"]
        component_rewards = type_cast(
            Dict[str, jax.Array], batch_dict.get("component_rewards", {})
        )

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)
        rngs = {"dropout": dropout_key} if dropout_key is not None else {}

        # Policy forward: returns (logits, loss); we ignore loss.
        mutable = ["scalars"]
        if intermediates:
            mutable.extend(("intermediates", "plots"))
        (policy_logits, _), mutated = cstate.apply_fn(
            {"params": params},
            x,
            y,
            padding_mask=padding_mask,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
        )

        # Reference forward (frozen): reuse the same apply_fn with state.base.
        ref_logits, _ = cstate.apply_fn(
            {"params": cstate.base},
            x,
            y,
            padding_mask=padding_mask,
            deterministic=True,
        )
        ref_logits = jax.lax.stop_gradient(ref_logits)

        # log π(y_t | x) for both policies at every action position.
        policy_logp = jax.nn.log_softmax(policy_logits.astype(jnp.float32), axis=-1)
        new_logp = jnp.take_along_axis(policy_logp, y[..., None], axis=-1).squeeze(-1)

        ref_logp = jax.nn.log_softmax(ref_logits.astype(jnp.float32), axis=-1)
        ref_logp = jnp.take_along_axis(ref_logp, y[..., None], axis=-1).squeeze(-1)

        action_mask_f = action_mask.astype(jnp.float32)
        n_actions = jnp.maximum(action_mask_f.sum(), 1.0)

        # Importance ratio π_new / π_old. Masked to 1.0 outside action
        # positions so the metric isn't misleading; loss-side masking via
        # action_mask_f below is what actually keeps gradients local to
        # generated tokens.
        log_ratio = new_logp - old_log_probs
        ratio = jnp.where(action_mask, jnp.exp(log_ratio), 1.0)

        # Advantages: use raw smeared rewards. (Subclasses normalize by group, etc.)
        advantages = per_token_rewards * action_mask_f

        # Clipped surrogate (PPO objective). Loss = -E[min(ratio*A, clip(ratio)*A)].
        clip_eps = cstate.clip_eps
        unclipped = ratio * advantages
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        surrogate = -jnp.minimum(unclipped, clipped)
        surrogate_loss = (surrogate * action_mask_f).sum() / n_actions

        # KL penalty against the frozen reference. We use Schulman's k3
        # estimator: kl ≈ exp(logr) - 1 - logr, where logr = log π_new - log π_ref.
        # Unlike the raw log-ratio (k1), k3 is always non-negative, so a
        # positive β actually penalizes divergence (k1 can flip sign on
        # sampled tokens and accidentally *reward* moving away from ref).
        log_ratio_ref = (ref_logp - new_logp) * action_mask_f  # log(π_ref/π_new)
        kl_per_tok = (jnp.exp(log_ratio_ref) - 1.0 - log_ratio_ref) * action_mask_f

        kl = kl_per_tok.sum() / n_actions

        loss = surrogate_loss + cstate.beta * kl

        metrics: Dict[str, Any] = scalar_metadata(mutated.get("scalars", {}))
        metrics.update(
            {
                "ppo/surrogate_loss": surrogate_loss,
                "ppo/kl": kl,
                "ppo/reward_mean": (per_token_rewards * action_mask_f).sum()
                / n_actions,
                "ppo/ratio_mean": (ratio * action_mask_f).sum() / n_actions,
                "ppo/advantage_mean": (advantages * action_mask_f).sum() / n_actions,
                "ppo/advantage_abs_mean": (jnp.abs(advantages) * action_mask_f).sum()
                / n_actions,
                "ppo/n_actions": n_actions,
            }
        )
        # Per-component reward means. These are logging-only rollout means;
        # they do not feed the objective and are not action-token weighted.
        # Each column is mostly-zero (only rollouts whose source is this
        # component have a non-zero entry), so a plain mean would dilute by
        # 1/k. Use the source-only mean instead — sum / count_of_nonzero —
        # which the default identity reward() makes safe (zeros remain zeros).
        for name, ctr in component_rewards.items():
            ctr_f = ctr.astype(jnp.float32)
            source_count = jnp.sum((ctr_f != 0.0).astype(jnp.float32))
            metrics[f"ppo/component/{name}/reward_mean"] = ctr_f.sum() / jnp.maximum(
                source_count, 1.0
            )

        if intermediates:
            metrics.update(
                {
                    "intermediates": mutated.get("intermediates", {}),
                    "plots": mutated.get("plots", {}),
                }
            )

        return policy_logits, loss, metrics


class BackbonedPPOTrainer(BackbonedTrainer, PPOTrainer[Module]):
    """PPO trainer that initializes from a pretrained HuggingFace backbone.

    Mirrors ``BackbonedContrastiveTrainer``: pulls in BackbonedTrainer's
    `model` and `initialize` (load HF architecture and weights) plus PPOTrainer's
    state/data/forward overrides. The MRO resolves cleanly because both
    parents only override disjoint methods.
    """

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # super() resolves to BackbonedTrainer (MRO), giving us the HF-style
        # config without MODEL.gather(); we then add PPO/RL configs.
        return super()._config() + [PPOConfig, *Evaluator.config(cls.RL_EVALUATION)]
