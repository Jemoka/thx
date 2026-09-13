"""SLURM availability inspection."""

import re
from dataclasses import dataclass

from loguru import logger

from theseus.execute.provider.utils import run


#### gpu availability ####


@dataclass(frozen=True)
class NodeAvailability:
    """Configured and currently unallocated GPUs on one SLURM node."""

    name: str
    capacity: int
    available: int


def available_gpus(
    partition: str,
    host: str,
    gpu_type: str | None = None,
    timeout: float | None = None,
) -> list[NodeAvailability]:
    """Return usable nodes in a partition, most free GPUs first."""
    result = run(
        f"sinfo -p {partition} --Node -h -O NodeList,Gres,GresUsed,StateCompact",
        host,
        timeout=timeout,
    )
    if not result.ok:
        logger.warning(
            "SLURM | could not inspect partition {}: {}", partition, result.stderr
        )
        return []

    available: list[NodeAvailability] = []
    for line in result.stdout.splitlines():
        line = line.rstrip()
        state = re.search(
            r"(idle|mix|alloc|allocated|drain|drng|down|comp|resv|unk|maint|planned)[*-]?$",
            line,
            re.IGNORECASE,
        )
        if state is None or state.group(1).lower() in {
            "down",
            "drain",
            "drng",
            "maint",
        }:
            continue

        fields = line[: state.start()].split()
        if len(fields) < 3:
            continue
        node, configured, allocated = fields[:3]

        typed = re.search(r"gpu:(\w+):(\d+)", configured)
        generic = re.search(r"gpu:(\d+)", configured)
        if typed is not None:
            configured_type = typed.group(1)
            total = int(typed.group(2))
        elif generic is not None:
            configured_type = None
            total = int(generic.group(1))
        else:
            continue
        if gpu_type and configured_type and configured_type != gpu_type:
            continue

        used = re.search(r"gpu(?::\w+)?:(\d+)", allocated)
        count = total - (int(used.group(1)) if used is not None else 0)
        available.append(NodeAvailability(node, total, max(count, 0)))

    return sorted(available, key=lambda item: item.available, reverse=True)


def partition_gpu_types(
    host: str,
    partitions: list[str] | None = None,
    timeout: float | None = None,
) -> dict[str, set[str]]:
    """Return the GPU types advertised by each requested partition."""
    result = run("sinfo --Node -h -o '%P|%G'", host, timeout=timeout)
    if not result.ok:
        logger.warning(
            "SLURM | could not inspect partitions on {}: {}", host, result.stderr
        )
        return {}

    requested = set(partitions) if partitions is not None else None
    types: dict[str, set[str]] = {}
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        partition, gres = line.split("|", 1)
        partition = partition.rstrip("*")
        if requested is not None and partition not in requested:
            continue

        types.setdefault(partition, set())
        typed = re.search(r"gpu:(\w+):\d+", gres)
        if typed is not None:
            types[partition].add(typed.group(1))
        elif re.search(r"gpu:\d+", gres):
            types[partition].add("__generic__")

    return types
