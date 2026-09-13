"""Narrow hardware solver for declarative executions."""

from dataclasses import dataclass

from loguru import logger

from theseus.base.hardware import HardwareResult
from theseus.execute.combobulator import Combobulation
from theseus.execute.config import DispatchConfig
from theseus.execute.provider import (
    Provider,
    SlurmConfig,
    SlurmProvider,
    SSHConfig,
    SSHProvider,
    VolcanoConfig,
    VolcanoProvider,
)


@dataclass(frozen=True)
class SolveResult:
    """Hardware allocation and the provider that supplied it."""

    result: HardwareResult | None
    provider: Provider | None

    @property
    def ok(self) -> bool:
        """Return whether a provider supplied hardware."""
        return self.result is not None


def solve(
    execution: Combobulation,
    config: DispatchConfig | None = None,
    timeout: float = 30.0,
) -> SolveResult:
    """Find the first configured SSH or SLURM provider that can run an execution.

    Args:
        execution: Declarative execution and its resource constraints.
        config: Dispatch inventory. When omitted, load the standard XDG or
            legacy configuration.
        timeout: Timeout in seconds for each provider's remote availability checks.

    Returns:
        The selected hardware and provider, or an empty result when unsatisfied.
    """
    config = config or DispatchConfig.load()
    allowed = set(execution.preferred_clusters)
    excluded = set(execution.forbidden_clusters)
    if allowed & excluded:
        raise ValueError(
            f"Clusters both selected and excluded: {sorted(allowed & excluded)}"
        )
    unknown = (allowed | excluded) - config.clusters.keys()
    if unknown:
        raise ValueError(f"Unknown clusters: {sorted(unknown)}")
    ordered = [
        *config.priority,
        *(name for name in config.hosts if name not in config.priority),
    ]
    for name in ordered:
        host = config.hosts.get(name)
        if host is None:
            continue
        if allowed and host.cluster not in allowed:
            logger.debug("SOLVE | skipping {}: cluster not selected", name)
            continue
        if host.cluster in excluded:
            logger.debug("SOLVE | skipping {}: cluster excluded", name)
            continue
        provider: Provider
        if isinstance(host, SSHConfig):
            provider = SSHProvider(name, host)
        elif isinstance(host, SlurmConfig):
            provider = SlurmProvider(name, host)
        elif isinstance(host, VolcanoConfig):
            provider = VolcanoProvider(name, host)
        else:
            continue
        logger.debug("SOLVE | checking provider {} in cluster {}", name, host.cluster)
        result = provider.solve(execution, config, timeout)
        if result is not None:
            logger.info(
                "SOLVE | selected {}: {} GPUs across {} hosts",
                name,
                result.total_chips,
                len(result.hosts),
            )
            return SolveResult(result, provider)
        logger.debug("SOLVE | provider {} cannot satisfy this request", name)
    return SolveResult(None, None)
