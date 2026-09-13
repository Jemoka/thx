"""A learning-rate schedule factory paired with its configuration schema."""

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import optax

C = TypeVar("C")


@dataclass(frozen=True)
class Schedule(Generic[C]):
    config: type[C]
    schedule: Callable[[int, C], optax.Schedule]
