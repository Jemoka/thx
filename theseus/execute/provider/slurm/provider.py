"""SLURM execution provider."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from loguru import logger

from theseus.base import Chip, SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.execute.bootstrap import generate
from theseus.execute.combobulator import Combobulation
from theseus.execute.provider.base import Provider
from theseus.execute.provider.slurm import availability
from theseus.execute.provider.utils import (
    RunResult,
    ShipResult,
    copy,
    place,
    run,
    uv_groups_for_cpu,
)

if TYPE_CHECKING:
    from theseus.execute.config import DispatchConfig
    from theseus.execute.dispatch import DispatchSpec


#### configuration ####


@dataclass
class PartitionConfig:
    """One SLURM partition available through a login host."""

    name: str
    default: bool = False
    constraint: str | None = None


@dataclass
class SlurmConfig:
    """A SLURM cluster allocated through an SSH login host."""

    ssh: str
    cluster: str
    type: Literal["slurm"] = "slurm"
    partitions: list[PartitionConfig] = field(default_factory=list)
    account: str | None = None
    qos: str | None = None
    mem: str | None = None
    time: str = "14-0"
    exclude: list[str] = field(default_factory=list)
    uv_groups: list[str] = field(default_factory=list)
    chips: dict[str, int] | None = None
    cpu_partitions: list[str] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


#### provider ####


@dataclass(frozen=True)
class SlurmProvider(Provider):
    """Solve allocations through one SLURM login host."""

    name: str
    host: SlurmConfig

    def solve(
        self,
        execution: Combobulation,
        config: DispatchConfig,
        timeout: float = 30.0,
    ) -> HardwareResult | None:
        """Query the head node and return schedulable hardware."""
        minimum = execution.gpus
        if minimum is None:
            minimum = 0 if execution.cpus is not None and not execution.chips else 1
        names = execution.chips or tuple(config.gres_mapping)
        selected: Chip | None = None
        layout: tuple[int, ...] | None = None

        if minimum == 0:
            partitions = self.host.cpu_partitions or [
                partition.name for partition in self.host.partitions
            ]
            layout = (0,) if partitions else None
        else:
            for name in names:
                chip = SUPPORTED_CHIPS.get(name) if name else None
                if name and chip is None:
                    logger.warning("Unknown requested chip {}", name)
                    continue
                if self.host.chips is None:
                    limit = None
                elif name:
                    limit = self.host.chips.get(name, 0)
                else:
                    limit = sum(self.host.chips.values())
                if limit is not None and limit < minimum:
                    continue

                gres = config.gres_mapping.get(name) if name else None
                if name and gres is None:
                    continue
                detected = availability.partition_gpu_types(
                    self.host.ssh,
                    [partition.name for partition in self.host.partitions],
                    timeout,
                )
                eligible = [
                    partition
                    for partition in self.host.partitions
                    if (gres and gres in detected.get(partition.name, set()))
                    or (gres is None and detected.get(partition.name, set()))
                ]
                eligible.sort(key=lambda partition: not partition.default)

                queued: tuple[int, ...] | None = None
                for partition in eligible:
                    nodes = availability.available_gpus(
                        partition.name,
                        self.host.ssh,
                        gres,
                        timeout,
                    )
                    immediate = place(
                        minimum,
                        [node.available for node in nodes],
                        limit,
                    )
                    configured_layout = place(
                        minimum,
                        [node.capacity for node in nodes],
                        limit,
                    )
                    logger.debug(
                        "SLURM provider {} found layout {} now, {} queued in {}",
                        self.name,
                        immediate,
                        configured_layout,
                        partition.name,
                    )
                    if immediate is not None:
                        selected = chip
                        layout = immediate
                        break
                    if queued is None:
                        queued = configured_layout
                if layout is None and queued is not None:
                    logger.debug(
                        "SLURM provider {} found no immediately available allocation; "
                        "falling back to compatible queued capacity",
                        self.name,
                    )
                    selected = chip
                    layout = queued
                if layout is not None:
                    break

        if layout is None:
            return None
        configured = config.clusters[self.host.cluster]
        cluster = Cluster(
            name=self.host.cluster,
            root=configured.root,
            work=configured.work,
            log=configured.log,
            data=configured.data,
            checkpoints=configured.checkpoints,
            results=configured.results,
            objects=configured.objects,
            status=configured.status,
            mount=configured.mount,
            cache_size=configured.cache_size,
            cache_dir=configured.cache_dir,
            all_squash=configured.all_squash,
        )
        machines = [
            ClusterMachine(
                name=self.name,
                cluster=cluster,
                resources={selected: count} if selected is not None else {},
                uv_groups=(
                    list(self.host.uv_groups)
                    if selected is not None
                    else uv_groups_for_cpu(self.host.uv_groups)
                ),
                env=(
                    dict(self.host.env)
                    if selected is not None
                    else {
                        **self.host.env,
                        "CUDA_VISIBLE_DEVICES": "",
                        "JAX_PLATFORMS": "cpu",
                    }
                ),
            )
            for count in layout
        ]
        return HardwareResult(
            chip=selected,
            hosts=machines,
            total_chips=sum(layout),
        )

    def ship(
        self,
        spec: DispatchSpec,
        config: DispatchConfig,
        timeout: float = 30.0,
    ) -> ShipResult:
        """Publish a dispatch on shared storage and submit it once to SLURM."""
        if spec.hardware is None:
            raise ValueError("Remote dispatch requires resolved hardware")
        machines = spec.hardware.hosts
        if not machines:
            raise ValueError("Cannot ship a dispatch without machines")
        if any(machine.name != self.name for machine in machines):
            raise ValueError("Every SLURM machine must use the selected provider")
        if any(machine.cluster.name != self.host.cluster for machine in machines):
            raise ValueError("Every SLURM machine must use the provider cluster")

        cluster = machines[0].cluster
        configured_cluster = config.clusters[cluster.name]
        share = configured_cluster.share or f"{cluster.work}/.dispatch"
        directory = "/".join(
            (share.rstrip("/"), spec.project, spec.group, spec.name, spec.nonce)
        )
        bootstrap = f"{directory}/bootstrap.sh"
        dispatch = f"{directory}/dispatch.json"
        sbatch = f"{directory}/submit.sbatch"
        log_prefix = "-".join(
            part.replace("/", "_")
            for part in (spec.project, spec.group, spec.name, spec.nonce)
        )
        logs = tuple(
            f"{cluster.log_dir}/{log_prefix}.{index}.log"
            for index in range(len(machines))
        )

        if spec.hardware.chip is None:
            partitions = self.host.cpu_partitions or [
                partition.name for partition in self.host.partitions
            ]
        else:
            gres = config.gres_mapping.get(spec.hardware.chip.name)
            if gres is None:
                raise ValueError(f"No SLURM GRES maps chip {spec.hardware.chip.name!r}")
            detected = availability.partition_gpu_types(
                self.host.ssh,
                [partition.name for partition in self.host.partitions],
                timeout,
            )
            partitions = [
                partition.name
                for partition in sorted(
                    self.host.partitions,
                    key=lambda partition: not partition.default,
                )
                if gres in detected.get(partition.name, set())
            ]
        if not partitions:
            raise ValueError(
                f"No current SLURM partition serves {spec.hardware.chip or 'CPU'}"
            )

        with TemporaryDirectory(prefix="theseus-dispatch-") as temporary:
            staged = Path(temporary) / spec.nonce
            staged.mkdir(mode=0o700)
            (staged / "bootstrap.sh").write_text(generate())
            (staged / "bootstrap.sh").chmod(0o700)
            (staged / "dispatch.json").write_text(spec.model_dump_json(indent=2))
            (staged / "dispatch.json").chmod(0o600)
            (staged / "submit.sbatch").write_text(
                self._sbatch(spec, config, partitions[0], bootstrap, dispatch)
            )
            (staged / "submit.sbatch").chmod(0o700)
            logger.debug(
                "SLURM | uploading dispatch to {}:{}", self.host.ssh, directory
            )
            copied = copy(
                staged,
                self.host.ssh,
                directory,
                timeout=timeout,
            )
        if not copied.ok:
            return ShipResult(
                self.name,
                self.host.ssh,
                directory,
                bootstrap,
                dispatch,
                logs,
                (),
                copied,
            )

        executable = run(
            f"chmod u+x -- {shlex.quote(bootstrap)} {shlex.quote(sbatch)}",
            self.host.ssh,
            timeout=timeout,
        )
        if not executable.ok:
            return ShipResult(
                self.name,
                self.host.ssh,
                directory,
                bootstrap,
                dispatch,
                logs,
                (),
                executable,
            )

        marker = f"{directory}.job"
        lock = f"{marker}.lock"
        job_name = "-".join(
            part.replace("/", "_")
            for part in (spec.project, spec.group, spec.name, spec.nonce)
        )
        quoted_marker = shlex.quote(marker)
        quoted_lock = shlex.quote(lock)
        command = (
            f"if test -s {quoted_marker}; then "
            f"job_id=$(cat {quoted_marker}); "
            'if squeue -h -j "$job_id" | grep -q .; then '
            "printf '%s\\n' \"$job_id\"; exit 0; fi; fi; "
            f"if ! mkdir {quoted_lock}; then "
            "echo 'another dispatch is submitting this nonce' >&2; exit 75; fi; "
            f"trap 'rmdir {quoted_lock}' EXIT; "
            f"job_id=$(sbatch --parsable {shlex.quote(sbatch)}); "
            "job_id=${job_id%%;*}; "
            f"printf '%s\\n' \"$job_id\" > {quoted_marker}.partial; "
            f"mv -f -- {quoted_marker}.partial {quoted_marker}; "
            "printf '%s\\n' \"$job_id\""
        )
        logger.debug("SLURM | submitting {} on {}", spec.name, self.host.ssh)
        submitted = run(command, self.host.ssh, timeout=timeout, max_attempts=1)
        if not submitted.ok:
            verify = (
                f"if test -s {quoted_marker}; then "
                f"job_id=$(cat {quoted_marker}); "
                'if squeue -h -j "$job_id" | grep -q .; then '
                "printf '%s\\n' \"$job_id\"; exit 0; fi; fi; "
                f"squeue -h -n {shlex.quote(job_name)} -o '%A' | head -n 1"
            )
            verified = run(verify, self.host.ssh, timeout=timeout)
            submitted = (
                verified if verified.ok and verified.stdout.strip() else submitted
            )

        job_ids = (
            (submitted.stdout.strip().splitlines()[-1],)
            if submitted.ok and submitted.stdout.strip()
            else ()
        )
        if job_ids and not job_ids[0].isdigit():
            submitted = RunResult(
                1,
                submitted.stdout,
                f"SLURM returned an invalid job ID: {job_ids[0]!r}",
            )
            job_ids = ()
        return ShipResult(
            self.name,
            self.host.ssh,
            directory,
            bootstrap,
            dispatch,
            logs,
            job_ids,
            submitted,
        )

    def _sbatch(
        self,
        spec: DispatchSpec,
        config: DispatchConfig,
        partition: str,
        bootstrap: str,
        dispatch: str,
    ) -> str:
        """Render the scheduler wrapper for a resolved logical layout."""
        if spec.hardware is None:
            raise ValueError("Remote dispatch requires resolved hardware")
        machines = spec.hardware.hosts
        job_name = "-".join(
            part.replace("/", "_")
            for part in (spec.project, spec.group, spec.name, spec.nonce)
        )
        cluster = machines[0].cluster
        log_prefix = f"{cluster.log_dir}/{job_name}"
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={len(machines)}",
            f"#SBATCH --ntasks={len(machines)}",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --output=/dev/null",
            "#SBATCH --error=/dev/null",
        ]
        lines.append(f"#SBATCH --cpus-per-task={spec.job.cpus or 2}")
        memory = spec.job.minimum_memory
        if memory is None:
            memory = "64G"
        # SLURM uses M/G/T suffixes rather than Mi/Gi/Ti.
        memory = str(memory).removesuffix("B").removesuffix("i")
        lines.append(f"#SBATCH --mem={memory}")
        lines.append(f"#SBATCH --time={self.host.time}")
        if self.host.account:
            lines.append(f"#SBATCH --account={self.host.account}")
        if self.host.qos:
            lines.append(f"#SBATCH --qos={self.host.qos}")
        if self.host.exclude:
            lines.append(f"#SBATCH --exclude={','.join(self.host.exclude)}")
        selected = next(
            (
                candidate
                for candidate in self.host.partitions
                if candidate.name == partition
            ),
            None,
        )
        if selected is not None and selected.constraint:
            lines.append(f"#SBATCH --constraint={selected.constraint}")

        chip = spec.hardware.chip
        if chip is not None:
            counts = [machine.resources.get(chip, 0) for machine in machines]
            if not counts or any(count != counts[0] for count in counts):
                raise ValueError("SLURM requires a homogeneous GPU layout")
            gres = config.gres_mapping[chip.name]
            lines.append(f"#SBATCH --gres=gpu:{gres}:{counts[0]}")

        wait = "${THESEUS_SRUN_WAIT_SECONDS:-120}"
        task = (
            f"set -o pipefail; THESEUS_LOG_PREFIX={shlex.quote(log_prefix)}; "
            f"THESEUS_STDOUT_MANAGED=1 bash -l {shlex.quote(bootstrap)} "
            f"{shlex.quote(dispatch)} 2>&1 | "
            'tee -a "${THESEUS_LOG_PREFIX}.${SLURM_PROCID}.log"'
        )
        lines.extend(
            (
                "",
                "set -euo pipefail",
                "umask 077",
                f"mkdir -p -- {shlex.quote(cluster.log_dir)}",
                "",
                f'srun --wait="{wait}" --nodes={len(machines)} '
                '--cpus-per-task="${SLURM_CPUS_PER_TASK}" '
                f"--ntasks={len(machines)} --ntasks-per-node=1 "
                f"bash -c {shlex.quote(task)}",
            )
        )
        return "\n".join(lines) + "\n"
