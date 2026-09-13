"""
Things you can train/infer/etc.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Optional

import jax
from jax.experimental import multihost_utils
from pydantic import BaseModel, Field

from theseus.base.topology import Topology
from theseus.base.axis import ShardingPolicy
from theseus.base.hardware import HardwareResult, local


class JobSpec(BaseModel):
    """user provided specification for a job"""

    name: str = Field(description="name of the job, useful for logging, etc.")
    id: Optional[str] = Field(
        description="ID (such as for wandb) of the job, could be None", default=None
    )
    project: Optional[str] = Field(
        description="project this run belongs to", default="general"
    )
    group: Optional[str] = Field(
        description="group under the project this run belongs to", default="default"
    )


class ExecutionSpec(JobSpec):
    """actually allocated specification for a job"""

    execution_id: Optional[str] = None
    tag: Optional[str] = None
    topology: Optional[Topology] = Field(description="hardware topology", default=None)
    hardware: HardwareResult = Field(description="actual allocated hardware")
    distributed: bool = Field(
        description="whether or not the run is happening over multiple hosts"
    )

    def result_path(self, name: str | Path) -> Path:
        project = self.project or "general"
        group = self.group or "default"
        results_dir = self.hardware.hosts[jax.process_index()].cluster.results_dir
        return Path(results_dir) / project / group / self.name / Path(name)

    @contextmanager
    def result(
        self,
        name: str | Path,
        main_process_only: bool = False,
        mode: str = "w",
        encoding: str | None = "utf-8",
    ) -> Iterator[IO[str] | None]:
        path_name = Path(name)
        sync_suffix = "__".join(path_name.parts) if path_name.parts else "result"
        sync_name = f"result:{self.name}:{sync_suffix}"
        if main_process_only:
            multihost_utils.sync_global_devices(f"{sync_name}:pre")
            try:
                if jax.process_index() != 0:
                    yield None
                    return

                path = self.result_path(path_name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, mode, encoding=encoding) as f:
                    yield f
            finally:
                multihost_utils.sync_global_devices(f"{sync_name}:post")
            return

        path = self.result_path(path_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding=encoding) as f:
            yield f

    @classmethod
    def local(
        cls,
        root_dir: str,
        name: str = "local",
        project: str | None = None,
        group: str | None = None,
        shard: ShardingPolicy | None = None,
    ) -> "ExecutionSpec":
        hardware = local(root_dir, "-")
        if hardware.chip is None:
            topology = None
        else:
            topology = Topology.new(hardware.chip, shard=shard)
        spec = cls(
            name=name,
            project=project,
            group=group,
            hardware=hardware,
            distributed=False,
            topology=topology,
        )

        return spec


class _BaseJob(ABC):
    @abstractmethod
    def __call__(self, resume: bool = False) -> None:
        """We will call this function simultaneously on all hosts"""
        None
