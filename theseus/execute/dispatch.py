"""Serializable remote execution dispatch."""

from pathlib import Path
from os import PathLike, fspath
from typing import cast

from pydantic import BaseModel, Field

from theseus.base import ExecutionSpec
from theseus.base.hardware import HardwareResult
from theseus.execute.combobulator import Combobulation
from theseus.execute.config import DispatchConfig
from theseus.execute.provider import (
    ShipResult,
    SlurmConfig,
    SlurmProvider,
    SSHConfig,
    SSHProvider,
    VolcanoConfig,
    VolcanoProvider,
)


class DispatchSpec(BaseModel):
    """Portable execution and its resolved logical machine layout."""

    name: str
    group: str
    project: str
    job: Combobulation
    nonce: str = Field(default_factory=lambda data: data["job"].nonce)
    hardware: HardwareResult | None = None

    def run(self, root: str | PathLike[str]) -> bool:
        """Run locally, discovering hardware and storing results under root."""
        from theseus.execute.run.run import Runner

        Path(root).mkdir(parents=True, exist_ok=True)
        spec = ExecutionSpec.local(
            fspath(root),
            name=self.name,
            project=self.project,
            group=self.group,
            shard=self.job.sharding,
        )
        spec.execution_id = self.nonce
        return Runner.new(self, spec).run()

    def launch(self, config: DispatchConfig) -> ShipResult:
        """Select the configured provider and launch this dispatch."""
        if self.hardware is None or not self.hardware.hosts:
            raise ValueError("DispatchSpec.hardware must contain at least one host")

        provider_names = {machine.name for machine in self.hardware.hosts}
        configured = {name: config.hosts.get(name) for name in provider_names}
        missing = sorted(name for name, host in configured.items() if host is None)
        if missing:
            raise ValueError(f"Dispatch providers are not configured: {missing}")

        hosts = tuple(configured.values())
        if all(isinstance(host, VolcanoConfig) for host in hosts):
            if len(provider_names) != 1:
                raise ValueError("A Volcano dispatch must use one provider")
            name = next(iter(provider_names))
            return VolcanoProvider(name, cast(VolcanoConfig, configured[name])).ship(
                self, config
            )
        if all(isinstance(host, SlurmConfig) for host in hosts):
            if len(provider_names) != 1:
                raise ValueError("A SLURM dispatch must use one provider")
            name = next(iter(provider_names))
            return SlurmProvider(name, cast(SlurmConfig, configured[name])).ship(
                self, config
            )
        if all(isinstance(host, SSHConfig) for host in hosts):
            name = self.hardware.hosts[0].name
            return SSHProvider(name, cast(SSHConfig, configured[name])).ship(
                self, config
            )
        raise ValueError("A dispatch cannot mix allocation providers")
