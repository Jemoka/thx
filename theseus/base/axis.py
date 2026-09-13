from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Sequence

from jax.sharding import Mesh, NamedSharding
from jax.tree_util import PyTreeDef

from pydantic import BaseModel, ConfigDict, Field, model_validator

Axis = SimpleNamespace(
    # Data-parallel replicas; also partitions ZeRO state and FSDP parameter storage.
    BATCH="batch",
    # Tensor-parallel devices within each replica.
    SHARD="shard",
)

# A logical dimension name and its assigned physical mesh axis or axes.
Rule = tuple[str, str | tuple[str, ...] | None]


class ShardingPolicy(BaseModel):
    """Execution parallelism; TP divides each data-parallel replica."""

    # Reject unknown options and prevent mutation after configuration.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Number of tensor-parallel devices per replica; must divide devices per host.
    tp: int = Field(default=1, ge=1, strict=True)
    # Store parameters across the batch axis and gather them for model execution.
    fsdp: bool = False
    # Shard optimizer state across the batch axis; required when FSDP is enabled.
    zero: bool = True
    # Recompute all forward intermediates during backward, not just parameter views.
    activation_checkpointing: bool = False

    @model_validator(mode="after")
    def _fsdp_state(self) -> "ShardingPolicy":
        if self.fsdp and not self.zero:
            raise ValueError("FSDP requires optimizer-state sharding (zero=True)")
        return self


@dataclass(frozen=True)
class ShardingPlan:
    """Logical dimension mappings for tensor and optimizer-state parallelism.

    By default ZeRO uses the TP dimension choices on the batch mesh axis.
    Unannotated tensors remain replicated regardless of the rules.
    """

    # Logical-axis rules for tensor parallelism during model execution.
    tp: Sequence[Rule] = ()
    # Additional optimizer-state rules; None reuses TP axis choices on Axis.BATCH.
    zero: Sequence[Rule] | None = None

    # Complete TP rules, computed from the model declaration.
    _tp: tuple[Rule, ...] = field(init=False)
    # Complete optimizer/FSDP storage rules: TP plus batch-axis partitions.
    _tp_x_zero: tuple[Rule, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tp", tuple(self.tp))
        object.__setattr__(
            self,
            "zero",
            tuple(self.zero)
            if self.zero is not None
            else tuple(
                (name, Axis.BATCH if axis is not None else None)
                for name, axis in self.tp
            ),
        )

        object.__setattr__(self, "_tp", tuple(self.tp))
        tp, zero = dict(self.tp), dict(self.zero or ())
        rules: list[Rule] = []
        for name in dict.fromkeys((*tp, *zero)):
            axes: list[str] = []
            for axis in (tp.get(name), zero.get(name)):
                if axis is not None:
                    axes.extend((axis,) if isinstance(axis, str) else axis)
            assigned = tuple(dict.fromkeys(axes))
            rules.append(
                (name, assigned[0] if len(assigned) == 1 else assigned or None)
            )
        object.__setattr__(self, "_tp_x_zero", tuple(rules))


@dataclass(frozen=True)
class ParameterShardingContext:
    """Contracts for parameter sharding.

    The optimizer state is just split into a few pieces and that's that.
    But haha! FSDP + potential TP means that there's like three different
    shapes the parameters may take. We keep track of all of them here."""

    # What's the hardware mesh we are talking about
    mesh: Mesh
    # Everything below is a bunch of tuples, so we need to know what is what
    parameters: PyTreeDef

    # ok so what's up with the mapping/sharding business?
    # `mapping` is a tuple from logical to physical axes. this is useful
    # for *flax* to know how to shard during forward/backward passes since
    # we don't explicitly tell Jax what's going on.
    #
    # `sharding` is a tuple of NamedShardings which tells Jax literally
    # how to place some pytree's worth of Arrays.

    # what happens during forward and backwards
    parameter_fwdbwd_mapping: tuple[Rule, ...]
    # what happens when its just sitting in a state
    parameter_storage_sharding: tuple[NamedSharding, ...]
    # what happens when a gradient step is happening/being applied
    parameter_update_mapping: tuple[Rule, ...]
    parameter_update_sharding: tuple[NamedSharding, ...]
