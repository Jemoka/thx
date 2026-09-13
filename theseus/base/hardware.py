"""Cluster information."""

from collections import defaultdict
from pathlib import Path
import re
import socket
import subprocess
from typing import Annotated, Any, Optional, Self

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from theseus.base.chip import Chip, SUPPORTED_CHIPS


class Cluster(BaseModel):
    name: str
    root: str  # root directory of checkpoints, code, etc.
    work: str  # work directory (where mirrors will be copied)
    log: Optional[str] = None  # log directory (defaults to {work}/logs)
    data: Optional[str] = None  # data directory (defaults to {root}/data)
    checkpoints: Optional[str] = (
        None  # checkpoints directory (defaults to {root}/checkpoints)
    )
    results: Optional[str] = None  # results directory (defaults to {root}/results)
    status: Optional[str] = None  # status directory (defaults to {root}/status)
    objects: Optional[str] = None  # object store (defaults to {root}/objects)

    # These launch settings travel in DispatchSpec so a remote bootstrap never
    # needs to consult the dispatching machine's local cluster inventory.
    mount: Optional[str] = None
    cache_size: Optional[str] = None
    cache_dir: Optional[str] = None
    all_squash: Optional[str] = None

    @property
    def log_dir(self) -> str:
        """Log directory path, defaults to {work}/logs if not configured."""
        d = self.log if self.log else f"{self.work}/logs"
        return d

    @property
    def root_dir(self) -> Path:
        # panic if not exist
        root_dir = Path(self.root)
        if not root_dir.exists():
            raise ValueError(
                f"Cluster root directory does not exist: cluster={self.name}, root={root_dir}"
            )

        return root_dir

    @property
    def data_dir(self) -> Path:
        data_dir = Path(self.data) if self.data else self.root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @property
    def checkpoints_dir(self) -> Path:
        checkpoints_dir = (
            Path(self.checkpoints)
            if self.checkpoints
            else self.root_dir / "checkpoints"
        )
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return checkpoints_dir

    @property
    def results_dir(self) -> Path:
        results_dir = Path(self.results) if self.results else self.root_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    @property
    def objects_dir(self) -> Path:
        results_dir = Path(self.objects) if self.objects else self.root_dir / "objects"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    @property
    def status_dir(self) -> Path:
        status_dir = Path(self.status) if self.status else self.root_dir / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        return status_dir


class ClusterMachine(BaseModel):
    name: str
    cluster: Cluster
    resources: dict[Chip, int]

    # Host-specific runtime settings are part of the resolved allocation. They
    # must survive serialization alongside the resources they are required for.
    uv_groups: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @field_serializer("resources")
    def _serialize_resources(self, resources: dict[Chip, int]) -> dict[str, int]:
        return {chip.name: count for chip, count in resources.items()}

    @field_validator("resources", mode="before")
    @classmethod
    def _deserialize_resources(cls, resources: Any) -> Any:
        if not isinstance(resources, dict) or all(
            isinstance(chip, Chip) for chip in resources
        ):
            return resources

        resolved: dict[Chip, int] = {}
        for chip_name, count in resources.items():
            chip = SUPPORTED_CHIPS.get(chip_name)
            if chip is None:
                raise ValueError(f"Chip {chip_name!r} is not located on this instance")
            resolved[chip] = count
        return resolved


class HardwareRequest(BaseModel):
    """
    Minimal, intent-level hardware request.

    Assumptions:
      - Storage is uniform across hosts (distributed FS).
      - Interconnects / topology are cluster concerns.
    """

    chip: Annotated[
        Optional[Chip],
        Field(default=None, description="chip type (None means any chip / cpu mode)"),
    ]
    min_chips: Annotated[
        int,
        Field(
            ge=0,
            default=1,
            description="minimum number of chips requested (0 means CPU mode)",
        ),
    ]
    preferred_clusters: Annotated[
        list[str],
        Field(default_factory=list, description="clusters to prefer (by name)"),
    ]
    forbidden_clusters: Annotated[
        list[str],
        Field(default_factory=list, description="clusters to avoid (by name)"),
    ]

    @model_validator(mode="after")
    def _validate(self) -> "HardwareRequest":
        overlap = set(self.preferred_clusters) & set(self.forbidden_clusters)
        if overlap:
            raise ValueError(
                f"Clusters cannot be both preferred and forbidden: {sorted(overlap)}"
            )

        return self


class HardwareResult(BaseModel):
    """
    Concrete hardware allocation result.
    """

    chip: Annotated[
        Optional[Chip],
        Field(description="chip type"),
    ]
    hosts: Annotated[
        list[ClusterMachine],
        Field(description="allocated hosts"),
    ]
    total_chips: Annotated[
        int,
        Field(ge=0, description="total number of allocated chips"),
    ]

    def redacted(self) -> Self:
        """Return a detached allocation safe for display and logging."""
        result = self.model_copy(deep=True)
        for host in result.hosts:
            host.env = dict.fromkeys(host.env, "<redacted>")
            if host.cluster.mount is not None:
                host.cluster.mount = "<redacted>"
        return result


def match(name: str) -> Optional[Chip]:
    """Match a reported GPU model name to the supported chip catalog."""
    reported = re.sub(r"\b(?:nvidia|amd|intel|google)\b", "", name.lower())
    tokens = set(re.findall(r"[a-z0-9]+", reported))
    best_chip = None
    best_score = 0
    for key, chip in SUPPORTED_CHIPS.items():
        if key.startswith("tpu-") or key == "cpu":
            continue
        catalog = re.sub(
            r"\b(?:nvidia|amd|intel|google)\b", "", chip.display_name.lower()
        )
        parts = re.findall(r"[a-z0-9]+", catalog)
        model = next((part for part in parts if any(c.isdigit() for c in part)), None)
        if model is None or model not in tokens:
            continue
        score = sum(len(part) for part in set(parts) & tokens)
        if score > best_score:
            best_chip, best_score = chip, score
    return best_chip


def local(root_dir: str, work_dir: str) -> HardwareResult:
    """
    Detect hardware using JAX device information.
    Creates one ClusterMachine per host, with host index matching jax.process_index().
    Assumes paths are identical across all hosts.
    """
    import jax

    devices = jax.devices()
    process_count = jax.process_count()

    # Group devices by process index to count per host
    devices_per_host: dict[int, int] = defaultdict(int)
    chip: Chip

    if not devices:
        # No devices at all, fall back to CPU
        chip = SUPPORTED_CHIPS["cpu"]
        total_chips = 1
        devices_per_host[0] = 1
    else:
        for d in devices:
            devices_per_host[d.process_index] += 1

        first_device = devices[0]
        device_kind = getattr(first_device, "device_kind", "")
        matched_chip = None
        if first_device.platform == "cpu":
            matched_chip = SUPPORTED_CHIPS["cpu"]
        elif first_device.platform == "tpu":
            device_kind_lower = device_kind.lower()
            matched_chip = next(
                (
                    candidate
                    for key, candidate in SUPPORTED_CHIPS.items()
                    if key.startswith("tpu-")
                    and key.removeprefix("tpu-") in device_kind_lower
                ),
                Chip(
                    name=f"tpu-{device_kind_lower.replace(' ', '-')}",
                    display_name=f"Google {device_kind}",
                    memory=int(16 * 1024**3),
                ),
            )
        elif first_device.platform == "gpu":
            matched_chip = match(device_kind)

        if matched_chip is None and first_device.platform == "gpu":
            try:
                inspected = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                row = next(
                    line for line in inspected.stdout.splitlines() if line.strip()
                )
                _, gpu_name, memory = (part.strip() for part in row.split(",", 2))
                unknown_name = re.sub(
                    r"\b(?:nvidia|amd|intel|google)\b", "", gpu_name.lower()
                )
                matched_chip = match(gpu_name) or Chip(
                    name=re.sub(r"[^a-z0-9]+", "-", unknown_name).strip("-"),
                    display_name=gpu_name,
                    memory=int(memory) * 1024**2,
                )
            except (
                FileNotFoundError,
                StopIteration,
                subprocess.CalledProcessError,
                ValueError,
            ):
                pass

        # Final fallback to CPU if still no chip
        if matched_chip is None:
            chip = SUPPORTED_CHIPS["cpu"]
        else:
            chip = matched_chip

        total_chips = len(devices)

    # Create cluster (paths assumed identical across hosts)
    cluster = Cluster(name="local", root=root_dir, work=work_dir)

    # Get current host's actual hostname
    current_process_idx = jax.process_index()
    current_hostname = socket.gethostname()

    # Create one ClusterMachine per host
    # Host index matches jax.process_index()
    hosts: list[ClusterMachine] = []
    for host_idx in range(process_count):
        host_device_count = devices_per_host.get(host_idx, 0)
        resources: dict[Chip, int] = {}
        if host_device_count > 0:
            resources[chip] = host_device_count

        # Use actual hostname for current host, fallback to index for others
        if host_idx == current_process_idx:
            host_name = current_hostname
        else:
            host_name = f"host-{host_idx}"

        hosts.append(
            ClusterMachine(
                name=host_name,
                cluster=cluster,
                resources=resources,
            )
        )

    return HardwareResult(chip=chip, hosts=hosts, total_chips=total_chips)
