from .axis import Axis as Axis
from .axis import ShardingPolicy as ShardingPolicy
from .axis import ShardingPlan as ShardingPlan
from .topology import Topology as Topology
from .chip import Chip, TheoreticalFLOPS, SUPPORTED_CHIPS
from .job import _BaseJob, JobSpec, ExecutionSpec
from .hardware import local
from .dag import Node

from typing import TypeVar, TypeAlias

T = TypeVar("T")
PyTree: TypeAlias = (
    T | list["PyTree[T]"] | tuple["PyTree[T]", ...] | dict[str, "PyTree[T]"]
)

__all__ = [
    "Topology",
    "Axis",
    "ShardingPolicy",
    "ShardingPlan",
    "Chip",
    "TheoreticalFLOPS",
    "SUPPORTED_CHIPS",
    "_BaseJob",
    "JobSpec",
    "ExecutionSpec",
    "PyTree",
    "Node",
    "local",
]
