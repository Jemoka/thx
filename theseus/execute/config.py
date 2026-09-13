"""Configuration owned by the execution providers."""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Self, TypeVar, cast

from omegaconf import OmegaConf

from theseus.base import SUPPORTED_CHIPS
from theseus.execute.provider import (
    PartitionConfig,
    SlurmConfig,
    SSHConfig,
    VolcanoConfig,
)

T = TypeVar("T")


@dataclass
class ClusterConfig:
    """Filesystem and environment shared by a compute cluster."""

    root: str
    work: str
    log: str | None = None
    data: str | None = None
    checkpoints: str | None = None
    results: str | None = None
    objects: str | None = None
    status: str | None = None
    share: str | None = None
    mount: str | None = None
    cache_size: str | None = None
    cache_dir: str | None = None
    all_squash: str | None = None


@dataclass
class DispatchConfig:
    """SSH, SLURM, and Volcano inventory used to solve an execution.

    The default loader reads ``$XDG_CONFIG_HOME/theseus/config.yaml`` and
    falls back to the legacy ``~/.theseus.yaml`` location.
    """

    mount: str | None = None
    proxy: str | None = None
    clusters: dict[str, ClusterConfig] = field(default_factory=dict)
    hosts: dict[str, SSHConfig | SlurmConfig | VolcanoConfig] = field(
        default_factory=dict
    )
    priority: list[str] = field(default_factory=list)
    gres_mapping: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _hydrate(schema: type[T], values: Mapping[str, Any]) -> T:
        """Construct a schema from its recognized fields."""
        names = {definition.name for definition in fields(cast(Any, schema))}
        return schema(
            **{name: value for name, value in values.items() if name in names}
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> Self:
        """Load an explicit or standard dispatch YAML file.

        Args:
            path: Explicit YAML path. When omitted, use the XDG path and then
                the legacy home-directory path.

        Returns:
            Parsed SSH, SLURM, and Volcano configuration. Unsupported host types are
            ignored.

        Raises:
            FileNotFoundError: If no requested or standard file exists.
            ValueError: If mutually exclusive top-level options are present.
        """
        if path is None:
            xdg = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "theseus"
                / "config.yaml"
            )
            legacy = Path.home() / ".theseus.yaml"
            path = next(
                (candidate for candidate in (xdg, legacy) if candidate.exists()), None
            )
            if path is None:
                raise FileNotFoundError(
                    f"No dispatch config found at {xdg} or {legacy}"
                )

        raw = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load(path)))
        clusters = {
            name: cls._hydrate(ClusterConfig, values)
            for name, values in raw.get("clusters", {}).items()
        }
        hosts: dict[str, SSHConfig | SlurmConfig | VolcanoConfig] = {}
        for name, values in raw.get("hosts", {}).items():
            values = dict(values)
            if values.get("chips") is not None:
                values["chips"] = {
                    SUPPORTED_CHIPS[chip].name: count
                    for chip, count in values["chips"].items()
                }
            if values.get("type", "plain") == "plain":
                hosts[name] = cls._hydrate(SSHConfig, values)
            elif values.get("type") == "volcano":
                hosts[name] = cls._hydrate(VolcanoConfig, values)
            elif values.get("type") == "slurm":
                partitions = [
                    PartitionConfig(partition)
                    if isinstance(partition, str)
                    else cls._hydrate(PartitionConfig, partition)
                    for partition in values.get("partitions", [])
                ]
                hosts[name] = cls._hydrate(
                    SlurmConfig,
                    {**values, "partitions": partitions},
                )

        mount = raw.get("mount")
        proxy = raw.get("proxy")
        if mount and proxy:
            raise ValueError("dispatch mount and proxy are mutually exclusive")
        return cls(
            mount=mount,
            proxy=proxy,
            clusters=clusters,
            hosts=hosts,
            priority=list(raw.get("priority", [])),
            gres_mapping={
                SUPPORTED_CHIPS[chip].name: gres
                for chip, gres in raw.get("gres_mapping", {}).items()
            },
        )
