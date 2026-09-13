"""
warmup-stable-decay-stable schedule
Think of this as WSD and then you run finetuning, all in one schedule
"""

from dataclasses import dataclass
from theseus.config import field
from .base import Schedule

import optax


@dataclass
class WSDSConfig:
    lr: float = field("optimization/lr", default=3e-4)

    warmup_pct: float = field("optimization/warmup_pct", default=0.01)
    decay_pct: float = field("optimization/decay_pct", default=0.01)
    constant_pct: float = field("optimization/constant_pct", default=0.5)

    warmup_lr_mult: float = field("optimization/warmup_lr_mult", default=0.01)
    decay_lr_mult: float = field("optimization/decay_lr_mult", default=0.1)


def wsds(total_steps: int, cfg: WSDSConfig) -> optax._src.base.Schedule:
    warmup_steps = int(total_steps * cfg.warmup_pct)
    decay_steps = int(total_steps * cfg.decay_pct)
    second_stable_steps = int(total_steps * cfg.constant_pct)
    first_stable_steps = total_steps - warmup_steps - decay_steps - second_stable_steps

    warmup_schedule = optax.linear_schedule(
        init_value=cfg.lr * cfg.warmup_lr_mult,
        end_value=cfg.lr,
        transition_steps=warmup_steps,
    )
    first_stable_schedule = optax.constant_schedule(cfg.lr)
    decay_schedule = optax.linear_schedule(
        init_value=cfg.lr,
        end_value=cfg.lr * cfg.decay_lr_mult,
        transition_steps=decay_steps,
    )
    second_stable_schedule = optax.constant_schedule(cfg.lr * cfg.decay_lr_mult)

    return optax.join_schedules(
        schedules=[
            warmup_schedule,
            first_stable_schedule,
            decay_schedule,
            second_stable_schedule,
        ],
        boundaries=[
            warmup_steps,
            warmup_steps + first_stable_steps,
            warmup_steps + first_stable_steps + decay_steps,
        ],
    )


WSDS = Schedule(WSDSConfig, wsds)
