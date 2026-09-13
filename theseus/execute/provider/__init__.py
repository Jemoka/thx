"""Allocation providers supported by the execution API."""

from theseus.execute.provider.base import Provider
from theseus.execute.provider.slurm import PartitionConfig, SlurmConfig, SlurmProvider
from theseus.execute.provider.ssh import SSHConfig, SSHProvider
from theseus.execute.provider.utils import ShipResult

__all__ = [
    "PartitionConfig",
    "Provider",
    "VolcanoConfig",
    "VolcanoProvider",
    "SlurmConfig",
    "SlurmProvider",
    "SSHConfig",
    "SSHProvider",
    "ShipResult",
]

from theseus.execute.provider.volcano import VolcanoConfig, VolcanoProvider
