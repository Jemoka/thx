"""Cosine rewarming schedule (Gupta et al., 2023).

At each dataset boundary the learning rate decays to ``min_lr``, then
linearly warms back up to ``rewarm_lr`` over ``warmup_steps`` steps,
then cosine-decays to ``min_lr`` again at the next boundary.

The first segment uses ``max_lr`` as its peak (the initial warmup target).
Subsequent segments use ``rewarm_lr`` which is typically lower than
``max_lr``.

Boundary positions are derived from the per-stage token budgets in
``stage_tokens`` together with ``batch_size`` and ``block_size``.
"""

from dataclasses import dataclass
from typing import List

from theseus.config import field
from .base import Schedule

import optax


@dataclass
class CosineRewarmConfig:
    lr: float = field("optimization/lr", default=3e-4)
    rewarm_lr: float = field("optimization/rewarm_lr", default=1e-4)
    min_lr_mult: float = field("optimization/min_lr_mult", default=0.01)
    warmup_pct: float = field("optimization/warmup_pct", default=0.01)

    # Per-stage token budgets — used to compute boundary step positions.
    # Read from the same config key as ABCDConfig.total_tokens.
    stage_tokens: List[int] = field(
        "training/tokens",
        default_factory=lambda: [1_000_000_000],
    )
    batch_size: int = field("training/batch_size", default=512)
    block_size: int = field("architecture/block_size", default=2048)


def cosine_rewarm(
    total_steps: int, cfg: CosineRewarmConfig
) -> optax._src.base.Schedule:
    """Build a piecewise cosine-rewarm schedule across all stages.

    Stage 0: warmup(min_lr → max_lr) + cosine(max_lr → min_lr)
    Stage i>0: warmup(min_lr → rewarm_lr) + cosine(rewarm_lr → min_lr)

    Boundaries are computed from ``stage_tokens / (batch_size * block_size)``.
    """
    tokens_per_step = cfg.batch_size * cfg.block_size
    min_lr = cfg.lr * cfg.min_lr_mult

    # Compute per-stage step counts and cumulative boundary positions
    stage_steps = [max(1, tok // tokens_per_step) for tok in cfg.stage_tokens]

    schedules: list[optax._src.base.Schedule] = []
    boundaries: list[int] = []
    cursor = 0

    for i, steps in enumerate(stage_steps):
        peak_lr = cfg.lr if i == 0 else cfg.rewarm_lr
        warmup_steps = max(1, int(steps * cfg.warmup_pct))
        cosine_steps = steps - warmup_steps

        # Linear warmup: min_lr → peak_lr
        warmup = optax.linear_schedule(
            init_value=min_lr,
            end_value=peak_lr,
            transition_steps=warmup_steps,
        )
        schedules.append(warmup)

        if i > 0 or len(stage_steps) > 1:
            boundaries.append(cursor)
        cursor += warmup_steps

        # Cosine decay: peak_lr → min_lr
        cosine = optax.cosine_decay_schedule(
            init_value=peak_lr,
            alpha=cfg.min_lr_mult,
            decay_steps=max(1, cosine_steps),
        )
        schedules.append(cosine)
        boundaries.append(cursor)
        cursor += cosine_steps

    # Remove the first boundary if it's 0 (join_schedules expects
    # boundaries strictly between schedule segments)
    if boundaries and boundaries[0] == 0:
        boundaries = boundaries[1:]

    return optax.join_schedules(schedules=schedules, boundaries=boundaries)


CosineRewarm = Schedule(CosineRewarmConfig, cosine_rewarm)
