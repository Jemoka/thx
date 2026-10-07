"""GPU availability through the SSH provider's nvidia-smi inventory."""

import pytest

from theseus.execute.combobulator import Combobulation
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.provider import SSHConfig, SSHProvider
from theseus.execute.provider.utils import RunResult


@pytest.mark.parametrize("output,available", [
    ("NVIDIA H100, 0, 80000, GPU-0\n", True),
    ("NVIDIA H100, 4, 80000, GPU-0\n", True),
    ("NVIDIA H100, 38000, 80000, GPU-0\n", False),
    ("NVIDIA H100, 8000, 80000, GPU-0\n", False),
    ("NVIDIA H100, 7999, 80000, GPU-0\n", True),
    ("NVIDIA H100, N/A, 80000, GPU-0\n", False),
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


def test_ssh_solve_pins_free_matching_gpus(monkeypatch):
    host = SSHConfig(ssh="fake", cluster="test", chips={"h100": 2}, env={"CUDA_VISIBLE_DEVICES": "0"})
    config = DispatchConfig(clusters={"test": ClusterConfig(root="/root", work="/work")})
    output = (
        "NVIDIA H100, 38000, 80000, GPU-busy\n"
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, 97887, GPU-other\n"
        "NVIDIA H100, 0, 80000, GPU-first\n"
        "NVIDIA H100, 0, 80000, GPU-second\n"
        "NVIDIA H100, 0, 80000, GPU-third\n"
    )
    monkeypatch.setattr("theseus.execute.provider.ssh.run", lambda *args, **kwargs: RunResult(0, output, ""))
    result = SSHProvider("fake", host).solve(Combobulation(steps=(), gpus=2), config)
    assert result is not None
    assert result.hosts[0].env == {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}
