"""An optimizer factory paired with its configuration schema."""

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import optax

C = TypeVar("C")


@dataclass(frozen=True)
class Optimizer(Generic[C]):
    config: type[C]
    optimizer: Callable[[optax.Schedule | float, C], optax.GradientTransformation]
    # Approximate arithmetic per trainable parameter per update, including clipping.
    flops_per_param: float | None = None
