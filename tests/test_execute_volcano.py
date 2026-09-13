import json

import pytest

from theseus.execute.combobulator import Combobulation
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.provider import VolcanoConfig, VolcanoProvider
from theseus.execute.provider.utils import RunResult
from theseus.execute.provider.volcano.manifest import manifest
from theseus.execute.provider.volcano.transport import Kubernetes


@pytest.fixture
def inventory():
    host = VolcanoConfig(cluster="test", image="training:latest", pvc_name="shared",
                         chips={"h100": 8}, num_nodes=2, gpus_per_node=8)
    return DispatchConfig(clusters={"test": ClusterConfig(root="/workspace", work="/workspace/work")}, hosts={"volcano": host})


@pytest.mark.parametrize("gpus,nodes,per_node", [(0, 1, 0), (4, 1, 4), (12, 2, 6), (17, 0, 0)])
def test_volcano_allocation(inventory, monkeypatch, gpus, nodes, per_node):
    monkeypatch.setattr(Kubernetes, "run", lambda *args, **kwargs: RunResult(0, '{"status":{"state":"Open"}}', ""))
    provider = VolcanoProvider("volcano", inventory.hosts["volcano"])
    result = provider.solve(Combobulation(steps=(), gpus=gpus), inventory)
    if not nodes:
        assert result is None
        return
    assert len(result.hosts) == nodes
    assert result.total_chips == nodes * per_node
    assert all(sum(host.resources.values()) == per_node for host in result.hosts)
    if not gpus:
        assert result.hosts[0].env["JAX_PLATFORMS"] == "cpu"
        assert "cpu" in result.hosts[0].uv_groups


def test_volcano_launch_uses_execution_resources_and_rank(inventory, monkeypatch):
    commands = []
    published = []

    def run(self, *args, **kwargs):
        commands.append((args, kwargs))
        return RunResult(0, '{"status":{"state":"Open"}}', "")

    monkeypatch.setattr(Kubernetes, "run", run)
    monkeypatch.setattr(Kubernetes, "publish", lambda self, directory, files, **kwargs: published.append((directory, files)) or RunResult(0, "", ""))
    monkeypatch.setattr("theseus.execute.provider.volcano.generate", lambda: "#!/bin/bash\necho bootstrap")
    provider = VolcanoProvider("volcano", inventory.hosts["volcano"])
    execution = Combobulation(steps=(), gpus=12, cpus=10, minimum_memory=2048)
    hardware = provider.solve(execution, inventory)
    spec = DispatchSpec(name="train", project="project", group="group", nonce="abc123", hardware=hardware, job=execution.shard(tp=2))
    shipped = spec.launch(inventory)
    assert shipped.ok
    job = json.loads(commands[-1][1]["data"])
    task = job["spec"]["tasks"][0]
    pod = task["template"]["spec"]
    container = pod["containers"][0]
    assert task["replicas"] == job["spec"]["minAvailable"] == 2
    assert container["resources"]["requests"] == {"cpu": "10", "memory": "2048Mi", "nvidia.com/gpu": "6"}
    assert "VC_TASK_INDEX" in container["command"][-1]
    assert "/etc/volcano/worker.host" in container["command"][-1]
    assert "podAntiAffinity" in pod["affinity"]
    assert json.loads(published[0][1]["dispatch.json"])["hardware"]["total_chips"] == 12


def test_volcano_helper_is_removed_on_upload_failure(inventory, monkeypatch):
    calls = []

    def run(self, *args, **kwargs):
        calls.append(args)
        return RunResult(1 if args[0] == "exec" else 0, "", "upload failed")

    monkeypatch.setattr(Kubernetes, "run", run)
    result = Kubernetes(inventory.hosts["volcano"]).publish("/workspace/dispatch", {"dispatch.json": "{}"})
    assert not result.ok
    assert calls[-1][0] == "delete"
    assert "--wait=false" in calls[-1]


def test_volcano_manifests_have_bounded_helpers(inventory):
    host = inventory.hosts["volcano"]
    job = manifest(host, "loader", "sleep 600", {}, loader=True)
    pod = job["spec"]["tasks"][0]["template"]["spec"]
    assert pod["activeDeadlineSeconds"] == 600
    assert job["spec"]["maxRetry"] == 0
    assert pod["containers"][0]["image"] == "busybox:1.37"


@pytest.mark.parametrize("same_manifest", [True, False])
def test_volcano_repeated_launch_checks_manifest(inventory, monkeypatch, same_manifest):
    existing = {}

    def run(self, *args, **kwargs):
        if args[0] == "create":
            existing.update(json.loads(kwargs["data"]))
            if not same_manifest:
                existing["metadata"]["annotations"]["theseus.dev/dispatch-sha256"] = "different"
            return RunResult(1, "", "AlreadyExists")
        if args[1] == "jobs.batch.volcano.sh":
            return RunResult(0, json.dumps(existing), "")
        return RunResult(0, '{"status":{"state":"Open"}}', "")

    monkeypatch.setattr(Kubernetes, "run", run)
    monkeypatch.setattr(Kubernetes, "publish", lambda *args, **kwargs: RunResult(0, "", ""))
    monkeypatch.setattr("theseus.execute.provider.volcano.generate", lambda: "#!/bin/bash\ntrue")
    execution = Combobulation(steps=(), gpus=1)
    provider = VolcanoProvider("volcano", inventory.hosts["volcano"])
    dispatch = DispatchSpec(
        name="retry", project="test", group="test", nonce="abc123",
        hardware=provider.solve(execution, inventory), job=execution,
    )
    result = dispatch.launch(inventory)
    assert result.ok is same_manifest
    if not same_manifest:
        assert result.remote.stderr == "AlreadyExists"


def test_volcano_inventory_loading(tmp_path):
    path = tmp_path / "dispatch.yaml"
    path.write_text('''clusters:
  test:
    root: /workspace/results
    work: /workspace/work
hosts:
  batch:
    type: volcano
    cluster: test
    image: training:latest
    pvc_name: shared
    chips: {h100: 8}
    num_nodes: 2
priority: [batch]
''')
    inventory = DispatchConfig.load(path)
    host = inventory.hosts["batch"]
    assert isinstance(host, VolcanoConfig)
    assert host.num_nodes == 2
    assert host.image == "training:latest"
    assert inventory.priority == ["batch"]


@pytest.mark.parametrize("fail_sync", [0, 1, 2])
def test_pvc_publication_flushes_before_and_after_rename(inventory, monkeypatch, tmp_path, fail_sync):
    import shlex
    import subprocess

    destination = tmp_path / "published"
    events = tmp_path / "sync-events"
    commands = []

    def run(self, *args, **kwargs):
        commands.append(args)
        if args[0] != "exec":
            return RunResult(0, "", "")
        # Use the platform's mv without GNU's -T; the destination is absent.
        # Observe filesystem contents at each flush, including flush failures.
        shell = f'''
        mv() {{ command mv "$2" "$3"; }}
        count=0
        sync() {{
            count=$((count + 1))
            if test -f {shlex.quote(str(destination / 'dispatch.json'))}; then
                echo published >> {shlex.quote(str(events))}
            else
                echo staged >> {shlex.quote(str(events))}
            fi
            test "$count" -ne {fail_sync}
        }}
        ''' + args[-1]
        result = subprocess.run(["sh", "-c", shell], input=kwargs["data"], capture_output=True)
        return RunResult(result.returncode, result.stdout.decode(), result.stderr.decode())

    monkeypatch.setattr(Kubernetes, "run", run)
    result = Kubernetes(inventory.hosts["volcano"]).publish(str(destination), {"dispatch.json": "{}"})
    assert result.ok is (fail_sync == 0)
    assert events.read_text().splitlines() == (["staged"] if fail_sync == 1 else ["staged", "published"])
    assert destination.exists() is (fail_sync != 1)
    if destination.exists():
        assert destination.stat().st_mode & 0o777 == 0o700
        assert (destination / "dispatch.json").stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.partial-*"))
    assert commands[-1][0] == "delete"
