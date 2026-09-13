from dataclasses import dataclass
from theseus.config import field
from .base import Optimizer

import optax


@dataclass
class AdamWConfig:
    weight_decay: float = field("optimization/weight_decay", default=0.1)
    beta1: float = field("optimization/beta1", default=0.9)
    beta2: float = field("optimization/beta2", default=0.95)


def adamw(
    lr: optax._src.base.Schedule | float, cfg: AdamWConfig
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr,
            b1=cfg.beta1,
            b2=cfg.beta2,
            weight_decay=cfg.weight_decay,
        ),
    )


# Approximately 16 FLOPs for AdamW + parameter addition, and 4 for norm/clipping.
# Scalar schedule/counter work is omitted; sqrt/division count as one operation.
AdamW = Optimizer(AdamWConfig, adamw, flops_per_param=20.0)
