"""Volcano workload manifests."""

from typing import Any

from theseus.execute.provider.volcano.config import VolcanoConfig


def manifest(
    host: VolcanoConfig,
    name: str,
    command: str,
    resources: dict[str, dict[str, str]],
    replicas: int = 1,
    *,
    loader: bool = False,
) -> dict[str, Any]:
    """Build a gang-scheduled job for either publication or execution."""
    container: dict[str, Any] = {
        "name": "worker",
        "image": "busybox:1.37" if loader else host.image,
        "command": ["sh" if loader else "bash", "-c", command],
        "resources": resources,
        "volumeMounts": [{"name": "workspace", "mountPath": host.pvc_mount_path}],
    }
    pod: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {"name": "workspace", "persistentVolumeClaim": {"claimName": host.pvc_name}}
        ],
        "nodeSelector": host.node_selector,
        "tolerations": host.tolerations,
    }
    if host.service_account:
        pod["serviceAccountName"] = host.service_account
    if host.priority_class:
        pod["priorityClassName"] = host.priority_class
    if loader:
        pod["activeDeadlineSeconds"] = 600
    elif host.shm_size:
        pod["volumes"].append(
            {
                "name": "shm",
                "emptyDir": {"medium": "Memory", "sizeLimit": host.shm_size},
            }
        )
        container["volumeMounts"].append({"name": "shm", "mountPath": "/dev/shm"})
    # A logical machine must correspond to a different physical GPU host.
    if replicas > 1:
        pod["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchLabels": {"theseus.dev/dispatch": name}
                        },
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        }
    return {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": host.namespace, "labels": host.labels},
        "spec": {
            "schedulerName": "volcano",
            "queue": host.queue or "default",
            "minAvailable": replicas,
            "maxRetry": 0,
            "plugins": {"env": [], "svc": ["--publish-not-ready-addresses=true"]},
            "tasks": [
                {
                    "name": "worker",
                    "replicas": replicas,
                    "policies": [
                        {"event": "PodFailed", "action": "AbortJob"},
                        {"event": "PodEvicted", "action": "AbortJob"},
                        {"event": "TaskCompleted", "action": "CompleteJob"},
                    ],
                    "template": {
                        "metadata": {
                            "labels": {**host.labels, "theseus.dev/dispatch": name}
                        },
                        "spec": pod,
                    },
                }
            ],
        },
    }
