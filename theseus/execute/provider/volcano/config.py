"""Kubernetes inventory for Volcano allocations."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VolcanoConfig:
    """A homogeneous pool scheduled through a Volcano queue."""

    cluster: str
    image: str
    pvc_name: str
    type: Literal["volcano"] = "volcano"
    namespace: str = "default"
    queue: str = "default"
    pvc_mount_path: str = "/workspace"
    pvc_codedrop_path: str | None = None
    chips: dict[str, int] = field(default_factory=dict)
    num_nodes: int = 1
    gpus_per_node: int = 0
    gpu_resource_key: str = "nvidia.com/gpu"
    cpu: str | None = None
    memory: str | None = None
    cpu_cpu: str | None = None
    cpu_memory: str | None = None
    shm_size: str | None = None
    service_account: str | None = None
    priority_class: str | None = None
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, str]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    uv_groups: list[str] = field(default_factory=list)
    kubeconfig: str | None = None
    context: str | None = None
    rdma: bool = False
    rdma_per_node: int = 8
    helper_resources: dict[str, str] = field(
        default_factory=lambda: {
            "requests.cpu": "1",
            "requests.memory": "1Gi",
            "limits.cpu": "1",
            "limits.memory": "1Gi",
        }
    )
