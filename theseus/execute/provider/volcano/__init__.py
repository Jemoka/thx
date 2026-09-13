"""Gang-scheduled Kubernetes execution through Volcano."""

from __future__ import annotations

import json
import hashlib
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from loguru import logger

from theseus.base import SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.execute.bootstrap import generate
from theseus.execute.provider.base import Provider
from theseus.execute.provider.utils import (
    RunResult,
    ShipResult,
    place,
    uv_groups_for_cpu,
)
from theseus.execute.provider.volcano.config import VolcanoConfig
from theseus.execute.provider.volcano.manifest import manifest
from theseus.execute.provider.volcano.transport import Kubernetes

if TYPE_CHECKING:
    from theseus.execute.combobulator import Combobulation
    from theseus.execute.config import DispatchConfig
    from theseus.execute.dispatch import DispatchSpec

__all__ = ["VolcanoConfig", "VolcanoProvider"]


@dataclass(frozen=True)
class VolcanoProvider(Provider):
    """Resolve a homogeneous allocation and submit it to a Volcano queue."""

    name: str
    host: VolcanoConfig

    def solve(
        self, execution: Combobulation, config: DispatchConfig, timeout: float = 30.0
    ) -> HardwareResult | None:
        configured = config.clusters[self.host.cluster]
        logger.debug("VOLCANO | checking queue {}", self.host.queue or "default")
        queue = Kubernetes(self.host).run(
            "get",
            "queues.scheduling.volcano.sh",
            self.host.queue or "default",
            "-o",
            "json",
            timeout=timeout,
        )
        if (
            not queue.ok
            or json.loads(queue.stdout).get("status", {}).get("state") != "Open"
        ):
            logger.debug("VOLCANO | queue {} unavailable or not Open", self.host.queue)
            return None
        minimum = execution.gpus
        if minimum is None:
            minimum = 0 if execution.cpus is not None and not execution.chips else 1
        chip = None
        layout: tuple[int, ...] | None = (0,) if minimum == 0 else None
        for name in execution.chips or tuple(self.host.chips):
            if minimum == 0:
                break
            capacity = self.host.chips.get(name, 0)
            if self.host.gpus_per_node:
                capacity = min(capacity, self.host.gpus_per_node)
            layout = place(minimum, [capacity] * self.host.num_nodes)
            if layout is not None:
                chip = SUPPORTED_CHIPS[name]
                break
        if layout is None:
            return None
        cluster = Cluster(
            name=self.host.cluster,
            **{
                key: value
                for key, value in asdict(configured).items()
                if key in Cluster.model_fields
            },
        )
        env = dict(self.host.env)
        groups = list(self.host.uv_groups)
        if chip is None:
            env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
            groups = uv_groups_for_cpu(groups)
        return HardwareResult(
            chip=chip,
            total_chips=sum(layout),
            hosts=[
                ClusterMachine(
                    name=self.name,
                    cluster=cluster,
                    resources={chip: count} if chip else {},
                    uv_groups=groups,
                    env=env,
                )
                for count in layout
            ],
        )

    def ship(
        self, spec: DispatchSpec, config: DispatchConfig, timeout: float = 30.0
    ) -> ShipResult:
        if spec.hardware is None or not spec.hardware.hosts:
            raise ValueError("Volcano dispatch requires resolved hardware")
        machines = spec.hardware.hosts
        if any(
            machine.name != self.name or machine.cluster.name != self.host.cluster
            for machine in machines
        ):
            raise ValueError("Volcano machines must belong to the selected provider")
        if any(machine.resources != machines[0].resources for machine in machines):
            raise ValueError("Volcano requires homogeneous worker resources")
        if not self.host.image or not self.host.pvc_name:
            raise ValueError("Volcano requires an image and shared PVC")
        cluster = machines[0].cluster
        share = (
            self.host.pvc_codedrop_path
            or config.clusters[self.host.cluster].share
            or f"{cluster.work}/.dispatch"
        )
        directory = str(
            PurePosixPath(share) / spec.project / spec.group / spec.name / spec.nonce
        )
        if (
            not PurePosixPath(directory).is_relative_to(self.host.pvc_mount_path)
            or ".." in PurePosixPath(directory).parts
        ):
            raise ValueError(
                "Volcano dispatch directory must be inside the mounted PVC"
            )
        name = re.sub("[^a-z0-9-]", "-", f"theseus-{spec.name.lower()}")[:40].strip("-")
        name = f"{name}-{spec.nonce.lower()}"
        bootstrap, dispatch = f"{directory}/bootstrap.sh", f"{directory}/dispatch.json"
        transport = Kubernetes(self.host)
        published = transport.publish(
            directory,
            {
                "bootstrap.sh": generate(),
                "dispatch.json": spec.model_dump_json(indent=2),
            },
            timeout=max(timeout, 120),
        )
        if not published.ok:
            return ShipResult(
                self.name,
                self.host.context or "kubectl",
                directory,
                bootstrap,
                dispatch,
                (),
                (),
                published,
            )
        gpus = sum(machines[0].resources.values())
        memory = spec.job.minimum_memory
        if memory is None:
            memory = (self.host.cpu_memory if not gpus else self.host.memory) or "64Gi"
        resources = {
            "cpu": str(
                spec.job.cpus
                or (self.host.cpu_cpu if not gpus else self.host.cpu)
                or "1"
            ),
            "memory": f"{memory}Mi" if isinstance(memory, int) else memory,
        }
        if gpus:
            resources[self.host.gpu_resource_key] = str(gpus)
        if self.host.rdma:
            resources["rdma/rdma_shared_device_a"] = str(
                gpus or self.host.rdma_per_node
            )
        command = (
            'set -euo pipefail; export THESEUS_MACHINE_INDEX="${VC_TASK_INDEX}"; '
            f"export THESEUS_PROCESS_COUNT={len(machines)}; "
            'export THESEUS_COORDINATOR_ADDRESS="$(head -n 1 /etc/volcano/worker.host):29400"; '
            f"exec bash {shlex.quote(bootstrap)} {shlex.quote(dispatch)}"
        )
        job = manifest(
            self.host,
            name,
            command,
            {"requests": resources, "limits": resources},
            len(machines),
        )
        fingerprint = hashlib.sha256(
            json.dumps(job, sort_keys=True).encode()
        ).hexdigest()
        job["metadata"]["annotations"] = {"theseus.dev/dispatch-sha256": fingerprint}
        # create prevents an existing nonce from being silently changed/restarted.
        logger.debug("VOLCANO | submitting job {} with {} workers", name, len(machines))
        launched = transport.run(
            "create", "-f", "-", data=json.dumps(job).encode(), timeout=timeout
        )
        if not launched.ok:
            logger.debug(
                "VOLCANO | create failed; checking for an identical existing job {}",
                name,
            )
            existing = transport.run(
                "get", "jobs.batch.volcano.sh", name, "-o", "json", timeout=timeout
            )
            if (
                existing.ok
                and json.loads(existing.stdout)
                .get("metadata", {})
                .get("annotations", {})
                .get("theseus.dev/dispatch-sha256")
                == fingerprint
            ):
                launched = RunResult(0, name, "")
        return ShipResult(
            self.name,
            self.host.context or "kubectl",
            directory,
            bootstrap,
            dispatch,
            tuple(f"{name}-worker-{index}" for index in range(len(machines))),
            (name,) if launched.ok else (),
            launched,
        )
