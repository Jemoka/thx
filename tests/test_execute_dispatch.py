from dataclasses import dataclass
from pathlib import Path

import pytest

from theseus.base import SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.execute.combobulator import Combobulator
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.provider import (
    PartitionConfig,
    ShipResult,
    SlurmConfig,
    SlurmProvider,
    SSHConfig,
    SSHProvider,
)
from theseus.execute.provider.utils import RunResult
from theseus.job import BasicJob
from theseus.store import ValueRow


@dataclass
class DispatchJobConfig:
    pass


class DispatchJob(BasicJob[DispatchJobConfig]):
    @classmethod
    def config(cls) -> type[DispatchJobConfig]:
        return DispatchJobConfig

    def state_init(self) -> None:
        pass

    def state_restore(self, state: ValueRow) -> None:
        pass

    def run(self) -> None:
        pass


def dispatch_spec(
    tmp_path: Path,
    *machines: tuple[str, int],
) -> DispatchSpec:
    chip = SUPPORTED_CHIPS["h100"]
    cluster = Cluster(
        name="test",
        root=str(tmp_path / "root"),
        work="/cluster/work",
        log="/cluster/log",
    )
    hosts = [
        ClusterMachine(
            name=name,
            cluster=cluster,
            resources={chip: count},
            uv_groups=["cuda13"],
        )
        for name, count in machines
    ]
    return DispatchSpec(
        name="run",
        project="project",
        group="group",
        nonce="abc123",
        hardware=HardwareResult(
            chip=chip,
            hosts=hosts,
            total_chips=sum(count for _, count in machines),
        ),
        job=Combobulator().run(DispatchJob),
    )


def test_dispatch_launch_selects_ssh_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = dispatch_spec(tmp_path, ("plain", 2))
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(root=str(tmp_path / "root"), work="/cluster/work")
        },
        hosts={
            "plain": SSHConfig(
                ssh="plain.example",
                cluster="test",
                chips={"h100": 2},
            )
        },
    )
    expected = ShipResult(
        "plain",
        "plain.example",
        "/dispatch",
        "/dispatch/bootstrap.sh",
        "/dispatch/dispatch.json",
        ("/logs/0.log",),
        ("123",),
        RunResult(0, "123", ""),
    )
    monkeypatch.setattr(SSHProvider, "ship", lambda self, sent, loaded: expected)

    assert spec.launch(config) is expected


def test_dispatch_launch_selects_slurm_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = dispatch_spec(tmp_path, ("slurm", 4), ("slurm", 4))
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(root=str(tmp_path / "root"), work="/cluster/work")
        },
        hosts={
            "slurm": SlurmConfig(
                ssh="login.example",
                cluster="test",
                partitions=[PartitionConfig("gpu")],
            )
        },
    )
    expected = ShipResult(
        "slurm",
        "login.example",
        "/dispatch",
        "/dispatch/bootstrap.sh",
        "/dispatch/dispatch.json",
        ("/logs/0.log", "/logs/1.log"),
        ("456",),
        RunResult(0, "456", ""),
    )
    monkeypatch.setattr(SlurmProvider, "ship", lambda self, sent, loaded: expected)

    assert spec.launch(config) is expected


def test_ssh_ship_publishes_and_launches_one_machine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import theseus.execute.provider.ssh as ssh_provider

    spec = dispatch_spec(tmp_path, ("first", 2))
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(
                root=str(tmp_path / "root"),
                work="/cluster/work",
                share="/cluster/share",
            )
        },
        hosts={
            "first": SSHConfig("first.example", "test"),
        },
    )
    published: list[tuple[str, str, set[str]]] = []
    commands: list[tuple[str, str, int | None]] = []

    def fake_copy(source, host, destination, timeout=None):
        assert source.stat().st_mode & 0o777 == 0o700
        assert (source / "bootstrap.sh").stat().st_mode & 0o777 == 0o700
        assert (source / "dispatch.json").stat().st_mode & 0o777 == 0o600
        published.append((host, destination, {path.name for path in source.iterdir()}))
        return RunResult(0, "", "")

    def fake_run(command, host, timeout=None, max_attempts=None):
        commands.append((command, host, max_attempts))
        return RunResult(0, "100\n", "")

    monkeypatch.setattr(ssh_provider, "generate", lambda: "#!/usr/bin/env bash\n")
    monkeypatch.setattr(ssh_provider, "copy", fake_copy)
    monkeypatch.setattr(ssh_provider, "run", fake_run)

    result = SSHProvider("first", config.hosts["first"]).ship(spec, config)

    assert result.ok
    assert result.job_ids == ("100",)
    assert result.directory == "/cluster/share/project/group/run/abc123"
    assert result.logs == ("/cluster/log/project-group-run-abc123.0.log",)
    assert published == [
        (
            "first.example",
            result.directory,
            {"bootstrap.sh", "dispatch.json"},
        ),
    ]
    launch_commands = [command for command, _, attempts in commands if attempts == 1]
    assert len(launch_commands) == 1
    assert any(command.startswith("chmod u+x") for command, _, _ in commands)
    assert "THESEUS_MACHINE_INDEX=0" in launch_commands[0]
    assert "THESEUS_PROCESS_COUNT" not in launch_commands[0]
    assert "THESEUS_COORDINATOR_ADDRESS" not in launch_commands[0]
    assert "flock -x -o" in launch_commands[0]
    assert "umask 077" in launch_commands[0]
    assert ">> /cluster/log/project-group-run-abc123.0.log" in launch_commands[0]


def test_ssh_ship_rejects_multiple_machines(tmp_path: Path) -> None:
    spec = dispatch_spec(tmp_path, ("first", 2), ("second", 2))
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(root=str(tmp_path / "root"), work="/cluster/work")
        },
        hosts={"first": SSHConfig("first.example", "test")},
    )

    with pytest.raises(ValueError, match="exactly one machine"):
        SSHProvider("first", config.hosts["first"]).ship(spec, config)


def test_ssh_ship_preserves_an_unverified_launch_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import theseus.execute.provider.ssh as ssh_provider

    spec = dispatch_spec(tmp_path, ("plain", 2))
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(root=str(tmp_path / "root"), work="/cluster/work")
        },
        hosts={"plain": SSHConfig("plain.example", "test")},
    )

    monkeypatch.setattr(ssh_provider, "generate", lambda: "#!/usr/bin/env bash\n")
    monkeypatch.setattr(
        ssh_provider,
        "copy",
        lambda *_args, **_kwargs: RunResult(0, "", ""),
    )

    def fake_run(command, _host, timeout=None, max_attempts=None):
        if command.startswith("chmod"):
            return RunResult(0, "", "")
        if max_attempts == 1:
            return RunResult(1, "", "flock failed")
        return RunResult(1, "", "process not found")

    monkeypatch.setattr(ssh_provider, "run", fake_run)

    result = SSHProvider("plain", config.hosts["plain"]).ship(spec, config)

    assert not result.ok
    assert result.remote.stderr == "flock failed"


def test_slurm_ship_renders_ranked_launch_without_scheduler_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import theseus.execute.provider.slurm.provider as slurm_provider

    spec = dispatch_spec(tmp_path, ("slurm", 4), ("slurm", 4))
    config = DispatchConfig(
        clusters={
            "test": ClusterConfig(
                root=str(tmp_path / "root"),
                work="/cluster/work",
                share="/cluster/share",
            )
        },
        hosts={
            "slurm": SlurmConfig(
                ssh="login.example",
                cluster="test",
                partitions=[PartitionConfig("gpu", default=True)],
                account="research",
                mem="32G",
            )
        },
        gres_mapping={"h100": "h100"},
    )
    staged: dict[str, str] = {}
    commands: list[tuple[str, int | None]] = []

    def fake_copy(source, host, destination, timeout=None):
        assert source.stat().st_mode & 0o777 == 0o700
        assert (source / "bootstrap.sh").stat().st_mode & 0o777 == 0o700
        assert (source / "dispatch.json").stat().st_mode & 0o777 == 0o600
        assert (source / "submit.sbatch").stat().st_mode & 0o777 == 0o700
        staged.update(
            {path.name: path.read_text() for path in source.iterdir() if path.is_file()}
        )
        return RunResult(0, "", "")

    def fake_run(command, host, timeout=None, max_attempts=None):
        commands.append((command, max_attempts))
        if "sbatch --parsable" in command:
            return RunResult(0, "9012\n", "")
        return RunResult(0, "", "")

    monkeypatch.setattr(slurm_provider, "generate", lambda: "#!/usr/bin/env bash\n")
    monkeypatch.setattr(slurm_provider, "copy", fake_copy)
    monkeypatch.setattr(slurm_provider, "run", fake_run)
    monkeypatch.setattr(
        slurm_provider.availability,
        "partition_gpu_types",
        lambda *_args: {"gpu": {"h100"}},
    )

    result = SlurmProvider("slurm", config.hosts["slurm"]).ship(spec, config)

    assert result.ok
    assert result.job_ids == ("9012",)
    wrapper = staged["submit.sbatch"]
    assert "#SBATCH --nodes=2" in wrapper
    assert "#SBATCH --ntasks=2" in wrapper
    assert "#SBATCH --ntasks-per-node=1" in wrapper
    assert "#SBATCH --gres=gpu:h100:4" in wrapper
    assert "#SBATCH --output=/dev/null" in wrapper
    assert "#SBATCH --error=/dev/null" in wrapper
    assert "srun" in wrapper
    assert "set -o pipefail" in wrapper
    assert "umask 077" in wrapper
    assert "THESEUS_STDOUT_MANAGED=1" in wrapper
    assert "${SLURM_PROCID}.log" in wrapper
    assert "tee -a" in wrapper
    assert result.bootstrap in wrapper
    assert result.dispatch in wrapper
    assert any(command.startswith("chmod u+x") for command, _ in commands)
    assert any(attempts == 1 for _, attempts in commands)


@pytest.mark.parametrize(
    "memory, expected",
    [
        (None, "64G"),
        ("128G", "128G"),
        ("96Gi", "96G"),
        ("96GiB", "96G"),
        (2048, "2048"),
    ],
)
def test_slurm_uses_execution_resource_requirements(tmp_path, memory, expected):
    spec = dispatch_spec(tmp_path, ("slurm", 4))
    spec.job = spec.job.cpu(12)
    if memory is not None:
        spec.job = spec.job.memory(memory)
    host = SlurmConfig(
        ssh="login", cluster="test", partitions=[PartitionConfig("gpu")], mem="32G"
    )
    config = DispatchConfig(gres_mapping={"h100": "h100"})
    script = SlurmProvider("slurm", host)._sbatch(
        spec, config, "gpu", "/bootstrap.sh", "/dispatch.json"
    )
    assert f"#SBATCH --mem={expected}" in script
    assert "#SBATCH --cpus-per-task=12" in script
    # Site policy can adjust this without updating SLURM_TRES_PER_TASK.
    assert '--cpus-per-task="${SLURM_CPUS_PER_TASK}"' in script
    assert "#SBATCH --gres=gpu:h100:4" in script
