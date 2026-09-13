"""Single-host SSH allocation and dispatch."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from loguru import logger

from theseus.base import SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult, match
from theseus.execute.bootstrap import generate
from theseus.execute.combobulator import Combobulation
from theseus.execute.provider.base import Provider
from theseus.execute.provider.utils import (
    RunResult,
    ShipResult,
    copy,
    run,
    uv_groups_for_cpu,
)

if TYPE_CHECKING:
    from theseus.execute.config import DispatchConfig
    from theseus.execute.dispatch import DispatchSpec


#### configuration ####


@dataclass
class SSHConfig:
    """One machine available for direct execution over SSH."""

    ssh: str
    cluster: str
    type: Literal["plain"] = "plain"
    chips: dict[str, int] = field(default_factory=dict)
    uv_groups: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


#### provider ####


@dataclass(frozen=True)
class SSHProvider(Provider):
    """Allocate and launch work on exactly one SSH machine."""

    name: str
    host: SSHConfig

    def solve(
        self,
        execution: Combobulation,
        config: DispatchConfig,
        timeout: float = 30.0,
    ) -> HardwareResult | None:
        """Return this host when it alone satisfies the execution request."""
        configured = config.clusters.get(self.host.cluster)
        if configured is None:
            raise ValueError(
                f"SSH provider {self.name!r} references unknown cluster "
                f"{self.host.cluster!r}"
            )
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

        minimum = execution.gpus
        if minimum is None:
            minimum = 0 if execution.cpus is not None and not execution.chips else 1
        if minimum == 0:
            machine = ClusterMachine(
                name=self.name,
                cluster=cluster,
                resources={},
                uv_groups=uv_groups_for_cpu(self.host.uv_groups),
                env={
                    **self.host.env,
                    "CUDA_VISIBLE_DEVICES": "",
                    "JAX_PLATFORMS": "cpu",
                },
            )
            return HardwareResult(chip=None, hosts=[machine], total_chips=0)

        inspected = run(
            "nvidia-smi --query-gpu=name,memory.used,memory.total "
            "--format=csv,noheader,nounits",
            self.host.ssh,
            timeout=timeout,
        )
        if not inspected.ok:
            logger.warning(
                "Could not inspect GPUs on SSH provider {}: {}",
                self.name,
                inspected.stderr,
            )
            return None
        devices: list[tuple[str, int, int]] = []
        for line in inspected.stdout.splitlines():
            try:
                device, used, total = (value.strip() for value in line.rsplit(",", 2))
                devices.append(
                    (
                        device,
                        int(used),
                        int(total),
                    )
                )
            except ValueError:
                continue

        requested_names = execution.chips or tuple(self.host.chips)
        selected = None
        for name in requested_names:
            chip = SUPPORTED_CHIPS.get(name)
            if chip is None:
                logger.warning("Unknown requested chip {}", name)
                continue
            available = sum(
                match(device) == chip and used < max(total * 0.1, 1024)
                for device, used, total in devices
            )
            if min(available, self.host.chips.get(name, 0)) >= minimum:
                selected = chip
                break
        if selected is None:
            return None

        machine = ClusterMachine(
            name=self.name,
            cluster=cluster,
            resources={selected: minimum},
            uv_groups=list(self.host.uv_groups),
            env=dict(self.host.env),
        )
        return HardwareResult(
            chip=selected,
            hosts=[machine],
            total_chips=minimum,
        )

    def ship(
        self,
        spec: DispatchSpec,
        config: DispatchConfig,
        timeout: float = 30.0,
    ) -> ShipResult:
        """Publish one immutable dispatch and idempotently launch it over SSH."""
        if spec.hardware is None:
            raise ValueError("Remote dispatch requires resolved hardware")
        machines = spec.hardware.hosts
        if len(machines) != 1:
            raise ValueError("SSH dispatches require exactly one machine")
        machine = machines[0]
        if machine.name != self.name:
            raise ValueError("The SSH machine must use the selected provider")
        if machine.cluster.name != self.host.cluster:
            raise ValueError("The SSH machine must use the provider cluster")
        configured = config.clusters.get(self.host.cluster)
        if configured is None:
            raise ValueError(
                f"SSH provider {self.name!r} references unknown cluster "
                f"{self.host.cluster!r}"
            )

        cluster = machine.cluster
        share = configured.share or f"{cluster.work}/.dispatch"
        directory = "/".join(
            (share.rstrip("/"), spec.project, spec.group, spec.name, spec.nonce)
        )
        bootstrap = f"{directory}/bootstrap.sh"
        dispatch = f"{directory}/dispatch.json"
        log_stem = "-".join(
            part.replace("/", "_")
            for part in (spec.project, spec.group, spec.name, spec.nonce)
        )
        log = f"{cluster.log_dir}/{log_stem}.0.log"

        with TemporaryDirectory(prefix="theseus-dispatch-") as temporary:
            staged = Path(temporary) / spec.nonce
            staged.mkdir(mode=0o700)
            (staged / "bootstrap.sh").write_text(generate())
            (staged / "bootstrap.sh").chmod(0o700)
            (staged / "dispatch.json").write_text(spec.model_dump_json(indent=2))
            (staged / "dispatch.json").chmod(0o600)
            logger.debug("SSH | uploading dispatch to {}:{}", self.host.ssh, directory)
            published = copy(
                staged,
                self.host.ssh,
                directory,
                timeout=timeout,
            )
        if not published.ok:
            return ShipResult(
                self.name,
                self.host.ssh,
                directory,
                bootstrap,
                dispatch,
                (log,),
                (),
                published,
            )

        quoted_bootstrap = shlex.quote(bootstrap)
        executable = run(
            f"chmod u+x -- {quoted_bootstrap}",
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
                (log,),
                (),
                executable,
            )

        pattern = shlex.quote(f"[b]ash {re.escape(bootstrap)} {re.escape(dispatch)}")
        find_process = f"pgrep -f -- {pattern} | head -n 1"
        start = (
            f"pid=$({find_process}); "
            'if test -n "$pid"; then printf \'%s\\n\' "$pid"; exit 0; fi; '
            "setsid nohup env THESEUS_MACHINE_INDEX=0 THESEUS_STDOUT_MANAGED=1 "
            f"{quoted_bootstrap} {shlex.quote(dispatch)} "
            f">> {shlex.quote(log)} 2>&1 < /dev/null & "
            "printf '%s\\n' \"$!\""
        )
        command = (
            f"umask 077; mkdir -p -- {shlex.quote(cluster.log_dir)}; "
            f"flock -x -o -- {shlex.quote(directory)} bash -c {shlex.quote(start)}"
        )
        logger.debug("SSH | submitting {} on {}", spec.name, self.host.ssh)
        launched = run(command, self.host.ssh, timeout=timeout, max_attempts=1)
        if not launched.ok:
            verified = run(
                f'pid=$({find_process}); test -n "$pid" && printf \'%s\\n\' "$pid"',
                self.host.ssh,
                timeout=timeout,
            )
            if verified.ok and verified.stdout.strip():
                launched = verified

        job_id = launched.stdout.strip().splitlines()[-1] if launched.stdout else ""
        if launched.ok and not job_id.isdigit():
            launched = RunResult(
                1,
                launched.stdout,
                f"SSH launch returned an invalid PID: {job_id!r}",
            )
        job_ids = (job_id,) if launched.ok else ()
        return ShipResult(
            self.name,
            self.host.ssh,
            directory,
            bootstrap,
            dispatch,
            (log,),
            job_ids,
            launched,
        )
