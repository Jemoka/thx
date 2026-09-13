from dataclasses import dataclass
from typing import Any

import pytest
from omegaconf import OmegaConf

import theseus.registry as registry
from theseus.base import ShardingPolicy, ExecutionSpec, Node
from theseus.config import configuration, configure, current_config, field
from theseus.execute import Combobulation, DispatchSpec
from theseus.job import BasicJob
from theseus.quick import QuickQuery, init, quick
from theseus.store import ObjectStore


@dataclass
class QuickConfig:
    value: int = field("quick/value", default=1)


class FakeJob:
    @classmethod
    def config(cls) -> type[QuickConfig]:
        return QuickConfig

    @classmethod
    def local(
        cls,
        root_dir: str,
        base: Node | None = None,
        **kwargs: Any,
    ) -> "FakeJob":
        job = cls()
        job.root_dir = root_dir
        job.base = base
        job.kwargs = kwargs
        job.config_value = configure(QuickConfig).value
        job._resume_required = False
        job.called_resume: bool | None = None
        return job

    def setup(self, resume: bool = False) -> None:
        self.setup_resume = resume

    def __call__(self, resume: bool = False) -> bool:
        self.called_resume = resume
        return resume

    def finish(self) -> None:
        self.finished = True


class QuickSpecJob(FakeJob, BasicJob[QuickConfig]):
    run = FakeJob.__call__
    state_init = FakeJob.setup
    state_restore = FakeJob.setup


def register_fake_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "ensure_registered", lambda: None)
    monkeypatch.setitem(registry.JOBS, "tests/quickx", FakeJob)


def seed_nodes(root: str) -> list[Node]:
    hardware = ExecutionSpec.local(root).hardware
    store = ObjectStore(hardware)
    nodes = [
        Node(name="source", nonce="abcdef", seq=1),
        Node(name="source", nonce="abcdef", seq=3),
        Node(name="source", nonce="abcdef", seq=2),
    ]
    for node in nodes:
        with store.blob(
            node,
            {
                "_x_checkpoint": True,
                "_x_job": "tests/source",
                "eval/score": node.seq / 10,
            },
        ) as path:
            OmegaConf.save(
                OmegaConf.create({"quick": {"value": 5}}), path / "config.yaml"
            )
    store.close()
    return nodes


def test_quick_resumes_last_matching_node(tmp_path, monkeypatch) -> None:
    register_fake_job(monkeypatch)
    nodes = seed_nodes(str(tmp_path))

    with quick(tmp_path) as q:
        q.build("tests/quickx", "target")
        selected = q.find().name("source").nonce("abcdef").checkpoint().resume()
        q.config.quick.value = 7
        result = q()

    assert selected is q
    assert q.base == nodes[1]
    assert result is True
    assert q._instance.base == nodes[1]
    assert q._instance.config_value == 7
    assert q._instance._resume_required is True
    assert q._instance.called_resume is True


def test_init_branches_and_marks_created_job(tmp_path, monkeypatch) -> None:
    register_fake_job(monkeypatch)
    nodes = seed_nodes(str(tmp_path))
    q = init(tmp_path)
    q.build("tests/quickx", "target")
    try:
        query = q.find().name("source").nonce("abcdef")
        assert isinstance(query, QuickQuery)
        assert query.branch() is q

        job = q.create()

        assert q.base == nodes[1]
        assert job.base == nodes[1]
        assert job._resume_required is False
        assert job.setup_resume is False
        assert job() is False
    finally:
        q.close()


def test_repeated_find_builds_independent_queries(tmp_path, monkeypatch) -> None:
    register_fake_job(monkeypatch)
    nodes = seed_nodes(str(tmp_path))

    with quick(tmp_path) as q:
        q.build("tests/quickx", "target")
        first = q.find().name("missing")
        second = q.find().nonce("abcdef").job("tests/source")

        assert first is not second
        assert first.all() == []
        assert second.all() == sorted(nodes, key=lambda n: n.seq)
        assert q.find().seq(3).checkpoint().resume() is q
        assert q.base == nodes[1]


@pytest.mark.parametrize("ordering", ["sequence", "latest", "custom"])
def test_find_selection_order(tmp_path, monkeypatch, ordering) -> None:
    register_fake_job(monkeypatch)
    store = ObjectStore.local(str(tmp_path))
    older = Node(name="source", nonce="aaaaaa", seq=9)
    newer = Node(name="source", nonce="zzzzzz", seq=1)
    for node in (older, newer):
        with store.blob(node, {"_x_checkpoint": True}) as path:
            OmegaConf.save(
                OmegaConf.create({"quick": {"value": 5}}), path / "config.yaml"
            )
    store.close()

    with quick(tmp_path) as q:
        q.build("tests/quickx", "target")
        query = q.find().name("source").checkpoint()
        if ordering == "latest":
            query.latest()
        elif ordering == "custom":
            query.sort("_x_seq", ascending=False)
        query.resume()

    assert q.base == (older if ordering == "sequence" else newer)


def test_find_requires_a_match_before_creation(tmp_path, monkeypatch) -> None:
    register_fake_job(monkeypatch)
    seed_nodes(str(tmp_path))

    with quick(tmp_path) as q:
        q.build("tests/quickx", "target")
        assert not hasattr(q, "restore")
        with pytest.raises(LookupError, match="No nodes matched"):
            q.find().name("missing").resume()

        q.create()
        assert q.find().name("source").all()
        with pytest.raises(RuntimeError, match="before creating"):
            q.find().branch()


def test_query_only_session_and_build_guards(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("THESEUS_ROOT", str(tmp_path))
    before = current_config()
    with quick() as q:
        assert q.find().all() == []
        assert not (tmp_path / "objects").exists()
        assert current_config() is before
        with pytest.raises(RuntimeError, match="build"):
            _ = q.config
        with pytest.raises(RuntimeError, match="build"):
            q()
    q.close()
    assert current_config() is before


@pytest.mark.parametrize("selection", ["branch", "resume"])
def test_build_clears_selection_but_preserves_queries(tmp_path, monkeypatch, selection):
    register_fake_job(monkeypatch)
    nodes = seed_nodes(str(tmp_path))
    with quick(tmp_path) as q:
        getattr(q.find().name("source"), selection)()
        assert q.base == nodes[1]
        selected_query = q.find().name("source").seq(2)
        missing_query = q.find().name("missing")

        q.build(FakeJob, "target")
        assert q.base is None
        assert q._resume is False
        assert q.find().all() == sorted(nodes, key=lambda n: n.seq)
        assert missing_query.all() == []
        assert selected_query.branch() is q
        assert q.base == nodes[2]

        getattr(q.find().name("source"), selection)()
        instance = q.create()
        assert instance.base == nodes[1]
        assert instance._resume_required is (selection == "resume")


def test_identical_build_preserves_edits_selection_and_instance(tmp_path, monkeypatch):
    register_fake_job(monkeypatch)
    seed_nodes(str(tmp_path))
    overrides = OmegaConf.create({"quick": {"value": 3}})
    with quick(tmp_path) as q:
        q.build("tests/quickx", "target", config=overrides)
        q.find().name("source").resume()
        q.config.quick.value = 7
        instance = q.create()
        config = q.config

        assert q.build(FakeJob, "target", config=OmegaConf.create(overrides)) is q
        assert q.config is config
        assert current_config() is config
        assert q.create() is instance
        assert q.config.quick.value == 7
        assert q.base is instance.base
        assert q._resume is True
        assert not getattr(instance, "finished", False)

        # Mutating the caller's original config must count as changed arguments.
        overrides.quick.value = 9
        q.build(FakeJob, "target", config=overrides)
        assert instance.finished
        assert q.config.quick.value == 9
        assert q.base is None


@pytest.mark.parametrize("changed", ["job", "name", "project", "group", "config"])
def test_changed_build_closes_and_replaces_session(tmp_path, monkeypatch, changed):
    register_fake_job(monkeypatch)
    seed_nodes(str(tmp_path))

    class OtherJob(FakeJob):
        pass

    outer = OmegaConf.create({"outer": True})
    with configuration(outer):
        with quick(tmp_path) as q:
            q.build(FakeJob, "target")
            q.find().name("source").resume()
            first = q.create()
            args = {"job": FakeJob, "name": "target"}
            args[changed] = {
                "job": OtherJob,
                "name": "second",
                "project": "project",
                "group": "group",
                "config": OmegaConf.create({"quick": {"value": 8}}),
            }[changed]
            q.build(**args)
            assert first.finished
            assert q.base is None
            assert q._resume is False
            assert current_config() is q.config
            second = q.create()
            assert second is not first
            assert second.base is None
        assert second.finished
        assert current_config() is outer
        q.close()
        assert current_config() is outer


def test_close_and_exception_restore_context_and_allow_rebuild(tmp_path, monkeypatch):
    register_fake_job(monkeypatch)
    before = current_config()
    with pytest.raises(ValueError, match="failure"):
        with quick(tmp_path) as q:
            q.build(FakeJob, "target")
            first = q.create()
            raise ValueError("failure")
    assert first.finished
    assert current_config() is before
    with pytest.raises(RuntimeError, match="build"):
        _ = q.config
    try:
        q.build(FakeJob, "target")
        assert q.create() is not first
    finally:
        q.close()
    assert current_config() is before


@pytest.mark.parametrize("selection", ["branch", "resume"])
def test_build_without_job_restores_selected_node(tmp_path, monkeypatch, selection):
    from theseus.job import RestoreableJob
    from unittest.mock import Mock

    nodes = seed_nodes(str(tmp_path))
    instance = FakeJob()
    cfg = OmegaConf.create({"quick": {"value": 7}})
    restore = Mock(return_value=(instance, cfg))
    monkeypatch.setattr(RestoreableJob, "from_node", restore)
    before = current_config()
    with quick(tmp_path) as q:
        getattr(q.find().name("source"), selection)()
        assert q.config.quick.value == 5
        q.config.quick.value = 7
        assert q.build() is q
        assert q.base == nodes[1]
        assert q._resume == (selection == "resume")
        assert q.create() is instance
        assert instance.setup_resume == (selection == "resume")
        assert q.config is cfg
        assert current_config() is cfg
        assert q.build() is q
        restore.assert_called_once()
        assert restore.call_args.args[0] == nodes[1]
        assert restore.call_args.kwargs["resume"] == (selection == "resume")
        assert restore.call_args.kwargs["runtime_cfg"].quick.value == 7
        assert q() == (selection == "resume")
    assert instance.finished
    assert current_config() is before


def test_build_without_job_requires_selected_node(tmp_path):
    with quick(tmp_path) as q:
        with pytest.raises(RuntimeError, match="branch.*resume"):
            q.build()


def test_build_from_node_passes_overrides_and_keeps_session_on_failure(
    tmp_path, monkeypatch
):
    from theseus.job import RestoreableJob
    from unittest.mock import Mock

    register_fake_job(monkeypatch)
    seed_nodes(str(tmp_path))
    with quick(tmp_path) as q:
        q.build(FakeJob, "original")
        q.find().name("source").branch()
        original_config = q.config
        restore = Mock(side_effect=ValueError("missing saved config"))
        monkeypatch.setattr(RestoreableJob, "from_node", restore)
        overrides = OmegaConf.create({"quick": {"value": 9}})
        with pytest.raises(ValueError, match="missing saved config"):
            q.build(name="branch", config=overrides)
        assert q.config is original_config
        assert current_config() is original_config
        assert restore.call_args.args[1].name == "branch"
        assert restore.call_args.kwargs["runtime_cfg"] == overrides


@pytest.mark.parametrize("selection", ["branch", "resume"])
def test_checkpoint_selection_loads_config_before_build(
    tmp_path, monkeypatch, selection
):
    register_fake_job(monkeypatch)
    seed_nodes(str(tmp_path))
    outer = OmegaConf.create({"outer": True})
    with configuration(outer):
        with quick(tmp_path) as q:
            getattr(q.find().name("source"), selection)()
            assert q.config.quick.value == 5
            assert current_config() is q.config
            with pytest.raises(RuntimeError, match="build"):
                q.create()
            q.config.quick.value = 9
            # A new selection reloads the saved baseline rather than carrying edits.
            getattr(q.find().name("source"), selection)()
            assert q.config.quick.value == 5
        assert current_config() is outer


@pytest.mark.parametrize("saved", [None, "[1, 2]", "quick: [invalid yaml"])
def test_invalid_checkpoint_config_preserves_session(tmp_path, monkeypatch, saved):
    register_fake_job(monkeypatch)
    nodes = seed_nodes(str(tmp_path))
    bad = Node(name="bad", nonce="badcfg", seq=1)
    store = ObjectStore.local(str(tmp_path))
    with store.blob(bad, {"_x_checkpoint": True}) as path:
        if saved is not None:
            (path / "config.yaml").write_text(saved)
    store.close()
    with quick(tmp_path) as q:
        q.find().name("source").resume()
        cfg = q.config
        with pytest.raises(Exception):
            q.find().name("bad").branch()
        assert q.base == nodes[1]
        assert q._resume is True
        assert q.config is cfg
        assert current_config() is cfg


def test_call_reuses_prepared_job(tmp_path, monkeypatch):
    register_fake_job(monkeypatch)
    with quick(tmp_path) as q:
        q.build(FakeJob, "local")
        assert q() is False
        instance = q.create()
        assert q() is False
        assert q.create() is instance


def test_shard_configures_creation_and_rejects_live_changes(tmp_path, monkeypatch):
    register_fake_job(monkeypatch)
    with quick(tmp_path) as q:
        q.shard(tp=2, fsdp=True, activation_checkpointing=True).build(FakeJob, "sharded")
        instance = q.create()
        assert instance.kwargs["shard"] == ShardingPolicy(tp=2, fsdp=True, activation_checkpointing=True)
        assert q.shard(tp=2, fsdp=True, activation_checkpointing=True) is q
        with pytest.raises(RuntimeError, match="before creating"):
            q.shard(tp=4)
        assert q.create() is instance


@pytest.mark.parametrize("selection", [None, "branch", "resume"])
def test_spec_snapshots_session_for_dispatch(tmp_path, selection):
    nodes = seed_nodes(str(tmp_path))
    with quick(tmp_path) as q:
        q.build(QuickSpecJob, "export").shard(tp=2)
        if selection is not None:
            getattr(q.find().name("source"), selection)()
        q.config.quick.value = 7
        execution = q.spec()
        assert q._instance is None
        assert execution.jobs == (QuickSpecJob,)
        assert execution.sharding == ShardingPolicy(tp=2)
        assert execution.steps[0].resume == (selection == "resume")
        if selection is None:
            assert execution.steps[0].base is None
        else:
            assert q._store.query(execution.steps[0].base).all() == [nodes[1]]
        q.config.quick.value = 9
        assert execution.config.quick.value == 7
        execution.config.quick.value = 11
        assert q.config.quick.value == 9

    dispatch = DispatchSpec(
        name="export", project="tests", group="quick", job=execution.gpu(2)
    )
    restored = DispatchSpec.model_validate_json(dispatch.model_dump_json())
    assert restored.job.config.quick.value == 11
    assert restored.job.gpus == 2
    for snapshot in (restored.job, Combobulation.deserialize(execution.serialize())):
        assert snapshot.jobs == execution.jobs
        assert snapshot.steps[0].base == execution.steps[0].base
        assert snapshot.steps[0].resume == execution.steps[0].resume
    assert execution.branch(QuickSpecJob).steps[-1].base == "previous"


def test_spec_uses_restored_job_class(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from theseus.job import RestoreableJob

    seed_nodes(str(tmp_path))
    instance = QuickSpecJob.__new__(QuickSpecJob)
    monkeypatch.setattr(
        RestoreableJob,
        "from_node",
        Mock(return_value=(instance, OmegaConf.create({"quick": {"value": 8}}))),
    )
    with quick(tmp_path) as q:
        q.find().name("source").resume().build()
        assert q.spec().jobs == (QuickSpecJob,)
        assert q.spec().config.quick.value == 8
        assert q.spec().steps[0].resume


def test_spec_requires_build(tmp_path):
    seed_nodes(str(tmp_path))
    with quick(tmp_path) as q:
        with pytest.raises(RuntimeError, match="build"):
            q.spec()
        q.find().name("source").branch()
        with pytest.raises(RuntimeError, match="build"):
            q.spec()
        q.build(QuickSpecJob, "export")
    with pytest.raises(RuntimeError, match="build"):
        q.spec()


def test_checkpoint_selection_only_reads_blob_metadata(tmp_path):
    from unittest.mock import patch
    from theseus.store import QueryBuilder

    seed_nodes(str(tmp_path))
    with patch.object(
        QueryBuilder, "select", autospec=True, side_effect=QueryBuilder.select
    ) as select:
        with quick(tmp_path) as q:
            q.find().name("source").resume()
            assert q.config.quick.value == 5
        assert select.call_count == 1
        assert select.call_args.kwargs == {"keys": ["blob"]}
