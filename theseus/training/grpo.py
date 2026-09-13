"""GRPO trainer — PPO with group-relative advantage normalization.

Inherits PPOTrainer; overrides only the per-rollout-reward → per-token-reward
conversion to do group-relative standardization within fixed-size groups.
The flat list of rollouts is reshaped into (n_groups, group_size); within
each group the advantage is (r - mean) / (std + eps).
"""

from functools import cached_property
from dataclasses import dataclass
from typing import cast as type_cast
from typing import Any, List, Type, Generic

import numpy as np
from loguru import logger

from theseus.config import field, configure
from theseus.model.module import Module
from theseus.training.base import M
from theseus.training.ppo import PPOTrainer, BackbonedPPOTrainer


@dataclass
class GRPOConfig:
    group_size: int = field("optimization/grpo/group_size", default=4)


class GRPOTrainer(PPOTrainer[M], Generic[M]):
    """GRPO: PPO with group-relative advantage normalization."""

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [GRPOConfig]

    @cached_property
    def grpo_config(self) -> GRPOConfig:
        return type_cast(GRPOConfig, configure(GRPOConfig))

    def _samples_per_prompt(self) -> int:
        # Tells the rollout pipeline to draw `group_size` completions per prompt
        # so the buffer arrives as [p0_s0..p0_s(G-1), p1_s0..p1_s(G-1), ...].
        # Group-relative z-scoring below is only valid under that ordering.
        return int(self.grpo_config.group_size)

    def _smear_rewards(
        self, rewards: np.ndarray, action_mask: np.ndarray, discount: float
    ) -> np.ndarray:
        """Group-relative reward smearing.

        Reshapes the per-rollout reward array into (n_groups, group_size),
        standardizes within each group (z-score), then defers to PPO's reward-
        to-go smear for the per-token distribution.

        ORDERING CONTRACT (load-bearing — read before touching anything that
        produces or consumes `rewards`):
          The reshape (-1, g) is only valid if `rewards[i*g : (i+1)*g]` are G
          completions of the SAME prompt of the SAME component. Groups are
          intra-component: a group of G never crosses component boundaries
          because each component runs samples_per_prompt=G independently in
          RolloutEvaluation, producing N_i = (n_prompts_i * g) consecutive
          G-blocks; PPOTrainer._refill_buffer concatenates components in order
          (component 0's rollouts, then component 1's, ...), so the boundary
          between components always lands on a G-aligned offset.
          The ordering is established by _samples_per_prompt() → RolloutEvaluation
          duplicating each selected index G times consecutively, and preserved by:
            • RolloutEvaluation.__call__ (intermediates list built in index
              order)
            • Evaluator.evaluate (per-component intermediates list)
            • PPOTrainer._refill_buffer (component-by-component concatenation;
              contiguous host split via np.array_split; FIFO buffer extend)
            • PPOTrainer.batch (FIFO slice of the buffer)
          A shuffle anywhere along that path silently turns this z-score into
          noise — the assertion below catches divisibility violations but
          CANNOT detect a same-size shuffle. If you add a shuffle to any of
          those call sites, override _samples_per_prompt to return 1 first.
        """
        g = self.grpo_config.group_size
        if g <= 1:
            return super()._smear_rewards(rewards, action_mask, discount)

        B = rewards.shape[0]
        if B % g != 0:
            raise ValueError(
                f"GRPO | batch size {B} not divisible by group_size {g}; "
                f"check RLEvaluatorConfig.batch_size and accumulate_steps."
            )

        grouped = rewards.reshape(-1, g)
        mean = grouped.mean(axis=-1, keepdims=True)
        std = grouped.std(axis=-1, keepdims=True)
        adv = ((grouped - mean) / (std + 1e-8)).reshape(-1)
        logger.debug(
            "GRPO | normalized {} groups of {} (group_mean range [{:.3f},{:.3f}], group_std range [{:.3f},{:.3f}])",
            B // g,
            g,
            float(mean.min()),
            float(mean.max()),
            float(std.min()),
            float(std.max()),
        )

        return super()._smear_rewards(adv, action_mask, discount)


class BackbonedGRPOTrainer(BackbonedPPOTrainer, GRPOTrainer[Module]):
    """GRPO trainer that initializes from a pretrained HuggingFace backbone.

    Stacks BackbonedPPOTrainer (HF init + PPO state/forward) with GRPOTrainer
    (group-relative advantage normalization). MRO:
    BackbonedGRPOTrainer → BackbonedPPOTrainer → BackbonedTrainer → GRPOTrainer
    → PPOTrainer → BaseTrainer.
    """

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # super() resolves to BackbonedPPOTrainer, which gives the HF-style
        # config + PPOConfig + RLEvaluatorConfig. Add GRPOConfig on top.
        return super()._config() + [GRPOConfig]
