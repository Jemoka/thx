"""Resolve checkpoint lineage for declarative execution steps."""

from dataclasses import dataclass

from theseus.base import Node
from theseus.base.hardware import HardwareResult
from theseus.execute.combobulator import BaseOf
from theseus.store import ObjectStore


@dataclass(frozen=True)
class StepState:
    """Durable state already recorded for one execution step."""

    finished: bool
    checkpoint: Node | None


class Lineage:
    """Inspect and resolve durable execution lineage."""

    def __init__(self, hardware: HardwareResult, execution_id: str) -> None:
        self.hardware = hardware
        self.execution_id = execution_id

    def state(self, tag: str) -> StepState:
        """Return the finish tombstone and latest checkpoint for a step."""
        store = ObjectStore(self.hardware)
        try:
            finished = bool(
                store.query()
                .execution(self.execution_id)
                .tag(tag)
                .finished()
                .latest()
                .all()
            )
            checkpoints = (
                store.query()
                .execution(self.execution_id)
                .tag(tag)
                .checkpoint()
                .latest()
                .all()
            )
        finally:
            store.close()
        return StepState(
            finished=finished,
            checkpoint=checkpoints[0] if checkpoints else None,
        )

    def resolve(self, base: BaseOf, previous_tag: str | None) -> Node | None:
        """Return the first checkpoint matching a base rule, if one exists."""
        store = ObjectStore(self.hardware)
        try:
            if base == "previous":
                if previous_tag is None:
                    raise ValueError(
                        "The first execution step cannot use base='previous'"
                    )
                matches = (
                    store.query()
                    .execution(self.execution_id)
                    .tag(previous_tag)
                    .checkpoint()
                    .latest()
                    .all()
                )
            elif base is None:
                return None
            else:
                matches = store.query(base).checkpoint().all()
        finally:
            store.close()
        return matches[0] if matches else None
