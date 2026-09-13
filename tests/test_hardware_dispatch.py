"""Cluster directory configuration and native dispatch serialization."""

import pytest
from omegaconf import OmegaConf

from theseus.base.hardware import HardwareResult
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.combobulator import Combobulation
from theseus.execute.provider import SSHConfig, SSHProvider

@pytest.fixture
def make_hardware(tmp_path):
    """Build a minimal HardwareResult with all four overridable dirs set."""
    from theseus.base.chip import SUPPORTED_CHIPS
    from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult

    root = tmp_path / "root"
    root.mkdir()
    chip = SUPPORTED_CHIPS["h100"]
    cluster = Cluster(
        name="test",
        root=str(root),
        work=str(tmp_path / "work"),
        log=str(tmp_path / "log"),
        data=str(tmp_path / "data"),
        checkpoints=str(tmp_path / "ckpt"),
        results=str(tmp_path / "res"),
        status=str(tmp_path / "st"),
    )
    machine = ClusterMachine(name="host0", cluster=cluster, resources={chip: 4})
    return HardwareResult(chip=chip, hosts=[machine], total_chips=4)


class TestClusterDerivedDirs:
    def test_defaults_derive_from_root(self, tmp_path):
        from theseus.base.hardware import Cluster

        root = tmp_path / "r"
        root.mkdir()
        c = Cluster(name="x", root=str(root), work=str(tmp_path / "w"))
        assert c.data_dir == root / "data"
        assert c.checkpoints_dir == root / "checkpoints"
        assert c.results_dir == root / "results"
        assert c.status_dir == root / "status"
        # Each call auto-creates the dir.
        for d in (c.data_dir, c.checkpoints_dir, c.results_dir, c.status_dir):
            assert d.is_dir()
    def test_overrides_honored(self, tmp_path):
        from theseus.base.hardware import Cluster

        root = tmp_path / "r"
        root.mkdir()
        over = {
            "data": tmp_path / "od",
            "checkpoints": tmp_path / "ock",
            "results": tmp_path / "ore",
            "status": tmp_path / "os",
        }
        c = Cluster(
            name="x",
            root=str(root),
            work=str(tmp_path / "w"),
            data=str(over["data"]),
            checkpoints=str(over["checkpoints"]),
            results=str(over["results"]),
            status=str(over["status"]),
        )
        assert c.data_dir == over["data"]
        assert c.checkpoints_dir == over["checkpoints"]
        assert c.results_dir == over["results"]
        assert c.status_dir == over["status"]



def test_cluster_config_fields_default_none():
    cfg = ClusterConfig(root="/a", work="/b")
    assert cfg.data is cfg.checkpoints is cfg.results is cfg.status is None


def test_config_directories_survive_allocation_and_serialization(tmp_path):
    config_path = tmp_path / "dispatch.yaml"
    OmegaConf.save(OmegaConf.create({
        "clusters": {"c1": {"root": "/r", "work": "/w", "data": "/d", "checkpoints": "/ck", "results": "/re", "status": "/st"}},
        "hosts": {"host": {"type": "plain", "ssh": "login", "cluster": "c1"}},
    }), config_path)
    config = DispatchConfig.load(config_path)
    host = config.hosts["host"]
    assert isinstance(host, SSHConfig)
    hardware = SSHProvider("host", host).solve(Combobulation(steps=(), gpus=0), config)
    restored = HardwareResult.model_validate_json(hardware.model_dump_json())
    cluster = restored.hosts[0].cluster
    assert (cluster.data, cluster.checkpoints, cluster.results, cluster.status) == ("/d", "/ck", "/re", "/st")


def test_hardware_roundtrip(make_hardware):
    restored = HardwareResult.model_validate_json(make_hardware.model_dump_json())
    assert restored == make_hardware
