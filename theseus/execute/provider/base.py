"""Allocation provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from theseus.base.hardware import HardwareResult
from theseus.execute.provider.utils import ShipResult

if TYPE_CHECKING:
    from theseus.execute.combobulator import Combobulation
    from theseus.execute.config import DispatchConfig
    from theseus.execute.dispatch import DispatchSpec


class Provider(ABC):
    """One configured backend capable of allocating an execution."""

    name: str

    @abstractmethod
    def solve(
        self,
        execution: Combobulation,
        config: DispatchConfig,
        timeout: float = 30.0,
    ) -> HardwareResult | None:
        """Return hardware for an execution, or ``None`` when unavailable."""

    @abstractmethod
    def ship(
        self,
        spec: DispatchSpec,
        config: DispatchConfig,
        timeout: float = 30.0,
    ) -> ShipResult:
        """Publish and launch one dispatch."""
