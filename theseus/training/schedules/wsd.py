from dataclasses import dataclass
from theseus.config import field
from .base import Schedule

import optax


@dataclass
class WSDConfig:
    lr: float = field("optimization/lr", default=3e-4)
    warmup_pct: float = field("optimization/warmup_pct", default=0.005)
    decay_pct: float = field("optimization/decay_pct", default=0.01)
    final_lr_frac: float = field("optimization/final_lr_frac", default=0.01)


def wsd(total_steps: int, cfg: WSDConfig) -> optax._src.base.Schedule:
    warmup_steps = int(total_steps * cfg.warmup_pct)
    decay_steps = int(total_steps * cfg.decay_pct)
    stable_steps = total_steps - warmup_steps - decay_steps

    warmup_schedule = optax.linear_schedule(
        init_value=cfg.lr * cfg.final_lr_frac,
        end_value=cfg.lr,
        transition_steps=warmup_steps,
    )
    stable_schedule = optax.constant_schedule(cfg.lr)
    decay_schedule = optax.linear_schedule(
        init_value=cfg.lr,
        end_value=cfg.lr * cfg.final_lr_frac,
        transition_steps=decay_steps,
    )

    return optax.join_schedules(
        schedules=[warmup_schedule, stable_schedule, decay_schedule],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


WSD = Schedule(WSDConfig, wsd)
