"""GPU availability through the SSH provider's nvidia-smi inventory."""

import pytest

from theseus.execute.combobulator import Combobulation
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.provider import SSHConfig, SSHProvider
from theseus.execute.provider.utils import RunResult


@pytest.mark.parametrize("output,available", [
    ("NVIDIA H100, 0, 80000\n", True),
    ("NVIDIA H100, 4, 80000\n", True),
    ("NVIDIA H100, 38000, 80000\n", False),
    ("NVIDIA H100, 8000, 80000\n", False),
    ("NVIDIA H100, 7999, 80000\n", True),
    ("NVIDIA H100, N/A, 80000\n", False),
    ("NVIDIA H100, 0\n", False),
    ("", False),
])
def test_ssh_gpu_availability(monkeypatch, output, available):
    host = SSHConfig(ssh="fake", cluster="test", chips={"h100": 4})
    config = DispatchConfig(clusters={"test": ClusterConfig(root="/root", work="/work")})
    monkeypatch.setattr("theseus.execute.provider.ssh.run", lambda *args, **kwargs: RunResult(0, output, ""))
    result = SSHProvider("fake", host).solve(Combobulation(steps=(), gpus=1), config)
    assert (result is not None) is available


def test_ssh_inspection_failure(monkeypatch):
    host = SSHConfig(ssh="fake", cluster="test", chips={"h100": 4})
    config = DispatchConfig(clusters={"test": ClusterConfig(root="/root", work="/work")})
    monkeypatch.setattr("theseus.execute.provider.ssh.run", lambda *args, **kwargs: RunResult(1, "", "unreachable"))
    assert SSHProvider("fake", host).solve(Combobulation(steps=(), gpus=1), config) is None
