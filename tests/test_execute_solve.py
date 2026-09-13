from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from theseus.base import SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.config import field
from theseus.execute.combobulator import Combobulator
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.provider import (
    PartitionConfig,
    SlurmConfig,
    SlurmProvider,
    SSHConfig,
    SSHProvider,
)
from theseus.execute.solve import solve
from theseus.job import BasicJob


@dataclass
class SolveConfig:
    value: int = field("solve/value", default=1)


class SolveJob(BasicJob[SolveConfig]):
    @classmethod
    def config(cls) -> type[SolveConfig]:
        return SolveConfig


def hardware(tmp_path: Path) -> HardwareResult:
    chip = SUPPORTED_CHIPS["h100"]
    cluster = Cluster(name="test", root=str(tmp_path), work=str(tmp_path / "work"))
    machine = ClusterMachine(name="host", cluster=cluster, resources={chip: 2})
    return HardwareResult(chip=chip, hosts=[machine], total_chips=2)


def config_yaml(path: Path, host_type: str = "plain") -> None:
    host: dict[str, Any] = {
        "type": host_type,
        "ssh": "login",
        "cluster": "test",
    }
    if host_type == "plain":
        host["chips"] = {"h100": 2}
    else:
        host["partitions"] = ["gpu"]
    OmegaConf.save(
        {
            "clusters": {"test": {"root": "/root", "work": "/work"}},
            "hosts": {"login": host},
        },
        path,
    )


def test_dispatch_config_loads_xdg_then_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    xdg_path = xdg / "theseus" / "config.yaml"
    xdg_path.parent.mkdir(parents=True)
    config_yaml(xdg_path)
    config_yaml(tmp_path / ".theseus.yaml", host_type="slurm")

    assert isinstance(DispatchConfig.load().hosts["login"], SSHConfig)

    xdg_path.unlink()
    assert isinstance(DispatchConfig.load().hosts["login"], SlurmConfig)


def test_dispatch_config_canonicalizes_chip_aliases(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.yaml"
    OmegaConf.save(
        {
            "clusters": {"test": {"root": "/root", "work": "/work"}},
            "hosts": {
                "login": {
                    "type": "plain",
                    "ssh": "login",
                    "cluster": "test",
                    "chips": {"a100": 2},
                }
            },
            "gres_mapping": {"a100": "a100"},
        },
        path,
    )

    config = DispatchConfig.load(path)

    assert config.hosts["login"].chips == {"a100-sxm4-80gb": 2}
    assert config.gres_mapping == {"a100-sxm4-80gb": "a100"}


def test_solve_preserves_mixed_host_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = hardware(tmp_path)
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={
            "ssh": SSHConfig(ssh="ssh", cluster="test", chips={"h100": 2}),
            "slurm": SlurmConfig(
                ssh="slurm",
                cluster="test",
                partitions=[PartitionConfig("gpu")],
            ),
        },
        priority=["ssh", "slurm"],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        SSHProvider,
        "solve",
        lambda self, *_args, **_kwargs: calls.append(self.name) or None,
    )
    monkeypatch.setattr(
        SlurmProvider,
        "solve",
        lambda self, *_args, **_kwargs: calls.append(self.name) or expected,
    )

    result = solve(Combobulator().run(SolveJob), config)

    assert result.result == expected
    assert isinstance(result.provider, SlurmProvider)
    assert calls == ["ssh", "slurm"]


def test_solve_loads_default_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"ssh": SSHConfig(ssh="ssh", cluster="test", chips={"h100": 2})},
    )
    expected = hardware(tmp_path)
    monkeypatch.setattr(DispatchConfig, "load", lambda: config)
    monkeypatch.setattr(SSHProvider, "solve", lambda *_args, **_kwargs: expected)

    result = solve(Combobulator().run(SolveJob))

    assert result.result == expected
    assert isinstance(result.provider, SSHProvider)


def test_ssh_provider_translates_execution_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.ssh as ssh_provider

    expected = hardware(tmp_path)
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(
                root=str(tmp_path),
                work="/work",
                objects="/shared/objects",
                mount="redis://juicefs.example/1",
                cache_size="1024",
                cache_dir="/cache/juicefs",
                all_squash="1000:1000",
            )
        },
        hosts={
            "ssh": SSHConfig(
                ssh="ssh",
                cluster="test",
                chips={"h100": 2},
                uv_groups=["cuda"],
                env={"UV_CACHE_DIR": "/cache/uv"},
            )
        },
    )
    calls: list[tuple[str, str, float]] = []

    def remote_run(command: str, host: str, timeout: float) -> Any:
        calls.append((command, host, timeout))
        return type(
            "RemoteResult",
            (),
            {
                "ok": True,
                "stdout": "NVIDIA H100, 0, 80000\nNVIDIA H100, 0, 80000",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(ssh_provider, "run", remote_run)
    execution = Combobulator().run(SolveJob).gpu(2).chip("h100")

    provider = SSHProvider("ssh", config.hosts["ssh"])
    result = provider.solve(
        execution,
        config,
        timeout=12.0,
    )

    assert result is not None
    assert result.chip == expected.chip
    assert result.total_chips == 2
    assert result.hosts[0].cluster.objects == "/shared/objects"
    assert result.hosts[0].cluster.mount == "redis://juicefs.example/1"
    assert result.hosts[0].cluster.cache_size == "1024"
    assert result.hosts[0].cluster.cache_dir == "/cache/juicefs"
    assert result.hosts[0].cluster.all_squash == "1000:1000"
    assert result.hosts[0].uv_groups == ["cuda"]
    assert result.hosts[0].env == {"UV_CACHE_DIR": "/cache/uv"}
    assert calls == [
        (
            "nvidia-smi --query-gpu=name,memory.used,memory.total "
            "--format=csv,noheader,nounits",
            "ssh",
            12.0,
        )
    ]


def test_ssh_provider_does_not_combine_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.ssh as ssh_provider

    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={
            "first": SSHConfig("first", "test", chips={"h100": 2}),
            "second": SSHConfig("second", "test", chips={"h100": 2}),
        },
        priority=["first", "second"],
    )
    monkeypatch.setattr(
        ssh_provider,
        "run",
        lambda *_args, **_kwargs: type(
            "RemoteResult",
            (),
            {
                "ok": True,
                "stdout": "NVIDIA H100, 0, 80000\nNVIDIA H100, 0, 80000",
                "stderr": "",
            },
        )(),
    )

    result = SSHProvider("first", config.hosts["first"]).solve(
        Combobulator().run(SolveJob).gpu(3).chip("h100"),
        config,
    )

    assert result is None


def test_ssh_provider_does_not_substitute_a_different_free_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.ssh as ssh_provider

    host = SSHConfig("mixed", "test", chips={"h100": 1})
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"mixed": host},
    )
    monkeypatch.setattr(
        ssh_provider,
        "run",
        lambda *_args, **_kwargs: type(
            "RemoteResult",
            (),
            {
                "ok": True,
                "stdout": (
                    "NVIDIA H100 NVL, 70000, 95830\n"
                    "NVIDIA RTX PRO 6000 Blackwell, 0, 97887"
                ),
                "stderr": "",
            },
        )(),
    )

    result = SSHProvider("mixed", host).solve(
        Combobulator().run(SolveJob).gpu(1).chip("h100"),
        config,
    )

    assert result is None


def test_ssh_provider_matches_reported_blackwell_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.ssh as ssh_provider

    host = SSHConfig("mixed", "test", chips={"h100": 2, "b6000": 2})
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"mixed": host},
    )
    monkeypatch.setattr(
        ssh_provider,
        "run",
        lambda *_args, **_kwargs: type(
            "RemoteResult",
            (),
            {
                "ok": True,
                "stdout": (
                    "NVIDIA H100 NVL, 0, 95830\n"
                    "NVIDIA H100 NVL, 0, 95830\n"
                    "NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, 97887\n"
                    "NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, 97887"
                ),
                "stderr": "",
            },
        )(),
    )

    result = SSHProvider("mixed", host).solve(
        Combobulator().run(SolveJob).gpu(2).chip("b6000"),
        config,
    )

    assert result is not None
    assert result.chip == SUPPORTED_CHIPS["b6000"]


def test_ssh_cpu_layout_hides_unallocated_gpus(tmp_path: Path) -> None:
    host = SSHConfig(
        "plain",
        "test",
        chips={"h100": 2},
        uv_groups=["all", "cuda13"],
        env={"CUSTOM": "value"},
    )
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"plain": host},
    )

    result = SSHProvider("plain", host).solve(
        Combobulator().run(SolveJob).cpu(1),
        config,
    )

    assert result is not None
    assert result.hosts[0].uv_groups == ["all", "cpu"]
    assert result.hosts[0].env == {
        "CUSTOM": "value",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
    }


def test_slurm_provider_checks_head_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.slurm.availability as slurm_availability

    host = SlurmConfig(
        ssh="login",
        cluster="test",
        partitions=[PartitionConfig("gpu")],
        uv_groups=["cuda"],
        env={"UV_CACHE_DIR": "/cache/uv"},
    )
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(
                root=str(tmp_path),
                work="/work",
                objects="/shared/objects",
                mount="redis://juicefs.example/1",
            )
        },
        hosts={"slurm": host},
        gres_mapping={"h100": "h100"},
    )
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        slurm_availability,
        "partition_gpu_types",
        lambda ssh, _partitions, timeout: (
            calls.append((ssh, timeout)) or {"gpu": {"h100"}}
        ),
    )
    monkeypatch.setattr(
        slurm_availability,
        "available_gpus",
        lambda _partition, ssh, _gres, timeout: (
            calls.append((ssh, timeout))
            or [slurm_availability.NodeAvailability("node", 2, 2)]
        ),
    )

    result = SlurmProvider("slurm", host).solve(
        Combobulator().run(SolveJob).gpu(2).chip("h100"),
        config,
        timeout=8.0,
    )

    assert result is not None
    assert result.total_chips == 2
    assert result.hosts[0].cluster.objects == "/shared/objects"
    assert result.hosts[0].cluster.mount == "redis://juicefs.example/1"
    assert result.hosts[0].uv_groups == ["cuda"]
    assert result.hosts[0].env == {"UV_CACHE_DIR": "/cache/uv"}
    assert calls == [("login", 8.0), ("login", 8.0)]


def test_slurm_cpu_layout_selects_cpu_runtime_groups(tmp_path: Path) -> None:
    host = SlurmConfig(
        ssh="login",
        cluster="test",
        partitions=[PartitionConfig("gpu")],
        cpu_partitions=["cpu"],
        uv_groups=["all", "cuda13"],
        env={"CUSTOM": "value"},
    )
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"slurm": host},
    )

    result = SlurmProvider("slurm", host).solve(
        Combobulator().run(SolveJob).cpu(1),
        config,
    )

    assert result is not None
    assert result.hosts[0].uv_groups == ["all", "cpu"]
    assert result.hosts[0].env == {
        "CUSTOM": "value",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
    }


def test_slurm_provider_falls_back_to_queued_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.slurm.availability as slurm_availability

    host = SlurmConfig(
        ssh="login",
        cluster="test",
        partitions=[PartitionConfig("gpu")],
    )
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"slurm": host},
        gres_mapping={"h100": "h100"},
    )
    monkeypatch.setattr(
        slurm_availability,
        "partition_gpu_types",
        lambda *_args: {"gpu": {"h100"}},
    )
    monkeypatch.setattr(
        slurm_availability,
        "available_gpus",
        lambda *_args: [slurm_availability.NodeAvailability("node", 2, 1)],
    )

    result = SlurmProvider("slurm", host).solve(
        Combobulator().run(SolveJob).gpu(2).chip("h100"),
        config,
    )

    assert result is not None
    assert result.total_chips == 2


def test_slurm_provider_lowers_total_gpus_into_logical_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.slurm.availability as slurm_availability

    host = SlurmConfig(
        ssh="login",
        cluster="test",
        partitions=[PartitionConfig("gpu")],
    )
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"slurm": host},
        gres_mapping={"h100": "h100"},
    )
    monkeypatch.setattr(
        slurm_availability,
        "partition_gpu_types",
        lambda *_args: {"gpu": {"h100"}},
    )
    monkeypatch.setattr(
        slurm_availability,
        "available_gpus",
        lambda *_args: [
            slurm_availability.NodeAvailability("node-a", 6, 6),
            slurm_availability.NodeAvailability("node-b", 6, 6),
        ],
    )

    result = SlurmProvider("slurm", host).solve(
        Combobulator().run(SolveJob).gpu(8).chip("h100"),
        config,
    )

    chip = SUPPORTED_CHIPS["h100"]
    assert result is not None
    assert result.total_chips == 8
    assert [machine.resources for machine in result.hosts] == [
        {chip: 4},
        {chip: 4},
    ]


def test_slurm_provider_queues_the_smallest_configured_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.slurm.availability as slurm_availability

    host = SlurmConfig(
        ssh="login",
        cluster="test",
        partitions=[PartitionConfig("gpu")],
    )
    config = DispatchConfig(
        clusters={"test": ClusterConfig(root=str(tmp_path), work="/work")},
        hosts={"slurm": host},
        gres_mapping={"h100": "h100"},
    )
    monkeypatch.setattr(
        slurm_availability,
        "partition_gpu_types",
        lambda *_args: {"gpu": {"h100"}},
    )
    monkeypatch.setattr(
        slurm_availability,
        "available_gpus",
        lambda *_args: [
            slurm_availability.NodeAvailability("node-a", 8, 1),
            slurm_availability.NodeAvailability("node-b", 6, 1),
        ],
    )

    result = SlurmProvider("slurm", host).solve(
        Combobulator().run(SolveJob).gpu(8).chip("h100"),
        config,
    )

    chip = SUPPORTED_CHIPS["h100"]
    assert result is not None
    assert result.total_chips == 8
    assert [machine.resources for machine in result.hosts] == [{chip: 8}]


def test_slurm_availability_parses_free_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.slurm.availability as slurm_availability
    from theseus.execute.provider.utils import RunResult

    monkeypatch.setattr(
        slurm_availability,
        "run",
        lambda *_args, **_kwargs: RunResult(
            0,
            "node-a gpu:h100:8 gpu:h100:2(IDX:0-1) idle                 \n"
            "node-b gpu:h100:4 gpu:h100:4(IDX:0-3) alloc\n"
            "node-c gpu:h100:8 gpu:h100:0(IDX:N/A) down\n"
            "node-d gpu:h100:8 gpu:h100:8(IDX:0-7) mix-\n"
            "node-e gpu:h100:8 gpu:h100:0(IDX:N/A) drain-\n",
            "",
        ),
    )

    assert slurm_availability.available_gpus("gpu", "login", "h100") == [
        slurm_availability.NodeAvailability("node-a", 8, 6),
        slurm_availability.NodeAvailability("node-b", 4, 0),
        slurm_availability.NodeAvailability("node-d", 8, 0),
    ]


def test_slurm_availability_lists_partition_gpu_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import theseus.execute.provider.slurm.availability as slurm_availability
    from theseus.execute.provider.utils import RunResult

    monkeypatch.setattr(
        slurm_availability,
        "run",
        lambda *_args, **_kwargs: RunResult(
            0,
            "gpu*|gpu:h100:8\ncpu|(null)\nother|gpu:a100:4\n",
            "",
        ),
    )

    assert slurm_availability.partition_gpu_types("login", ["gpu", "cpu"]) == {
        "gpu": {"h100"},
        "cpu": set(),
    }
