import random
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, call

import jax
import numpy as np
import pytest

from theseus.base import ExecutionSpec, Node
from theseus.job import (
    BasicJob,
    CheckpointedJob,
    LoggingJob,
    RestoreableJob,
)
from theseus.store import ObjectStore, RecordStore


def test_tick_records_current_node_before_advancing() -> None:
    node = Node(name="train")
    store = Mock()
    job = SimpleNamespace(node=node, store=store, _setup_complete=True, _node_written=False)

    expected = node.next()
    BasicJob.tick(job)
    assert job.node is node

    store.value.assert_called_once_with(node, {})
    assert job.node == expected
    assert job._node_written is False


def test_tick_elides_empty_row_for_an_already_written_node() -> None:
    node = Node(name="train")
    store = Mock()
    job = SimpleNamespace(node=node, store=store, _setup_complete=True, _node_written=True)

    expected = node.next()
    BasicJob.tick(job)
    assert job.node is node

    store.value.assert_not_called()
    assert job.node == expected
    assert job._node_written is False


def test_call_honors_restore_time_resume_with_warning(monkeypatch) -> None:
    base = Node(name="train", nonce="abcdef", seq=3)
    store = Mock()
    store.query.return_value.node.return_value.select.return_value = [
        {"blob": "checkpoint"}
    ]
    warning = Mock()
    monkeypatch.setattr("theseus.job.logger.warning", warning)
    barrier = Mock()
    monkeypatch.setattr("theseus.job.multihost_utils.sync_global_devices", barrier)
    job = SimpleNamespace(
        _resume_required=True,
        _setup_complete=False,
        spec=SimpleNamespace(name="train"),
        base=base,
        node=Node(name="target"),
        store=store,
        state_restore=Mock(),
        _synchronize_node=Mock(side_effect=lambda node: node),
        run=Mock(),
        finish=Mock(),
    )

    job.setup = MethodType(BasicJob.setup, job)
    BasicJob.__call__(job)

    warning.assert_called_once()
    assert barrier.call_args_list == [call("train:start"), call("train:finish")]
    job._synchronize_node.assert_called_once_with(base)
    job.run.assert_called_once_with()
    job.finish.assert_called_once_with()
    job.state_restore.assert_called_once_with({"blob": "checkpoint"})
    assert job.node == base
    assert store.value.call_args_list == [
        call(base, {}),
        call(base, {"_x_finished": True}),
    ]


def test_synchronize_node_broadcasts_a_numeric_wire_format(monkeypatch) -> None:
    node = Node(name="train", nonce="abcdef", seq=3, parent="parent")
    broadcasts: list[np.ndarray] = []

    def broadcast(value, is_source):
        assert is_source
        array = np.asarray(value)
        broadcasts.append(array)
        return array

    monkeypatch.setattr(
        "theseus.job.multihost_utils.broadcast_one_to_all",
        broadcast,
    )
    job = SimpleNamespace(main_process=Mock(return_value=True))

    synchronized = BasicJob._synchronize_node(job, node)

    assert synchronized == node
    assert [array.dtype for array in broadcasts] == [
        np.dtype("int32"),
        np.dtype("uint8"),
    ]
    assert broadcasts[1].tobytes() == node.model_dump_json().encode()


def test_synchronize_node_receives_the_source_payload(monkeypatch) -> None:
    source = Node(name="train", nonce="abcdef", seq=3, parent="parent")
    payload = source.model_dump_json().encode()
    received = iter(
        (
            np.asarray(len(payload), dtype=np.int32),
            np.frombuffer(payload, dtype=np.uint8),
        )
    )

    monkeypatch.setattr(
        "theseus.job.multihost_utils.broadcast_one_to_all",
        lambda value, is_source: next(received),
    )
    job = SimpleNamespace(main_process=Mock(return_value=False))

    assert BasicJob._synchronize_node(job, Node(name="ignored")) == source


def test_log_normalizes_host_scalars() -> None:
    node = Node(name="train")
    store = Mock()
    job = SimpleNamespace(
        node=node,
        store=store,
        _node_written=False,
        _normalize_log=LoggingJob._normalize_log,
    )

    LoggingJob.log(
        job,
        {
            "python": 1,
            "numpy": np.float32(1.25),
            "jax": jax.numpy.asarray(2.5),
            "label": "train",
        },
    )

    store.value.assert_called_once_with(
        node,
        {"python": 1, "numpy": 1.25, "jax": 2.5, "label": "train"},
    )
    assert job._node_written is True


def test_log_rejects_non_scalar_values() -> None:
    job = SimpleNamespace(
        node=Node(name="train"),
        store=Mock(),
        _node_written=False,
        _normalize_log=LoggingJob._normalize_log,
    )

    with pytest.raises(ValueError, match="'losses'.*shape \\(2,\\)"):
        LoggingJob.log(job, {"losses": np.ones(2)})

    job.store.value.assert_not_called()
    assert job._node_written is False


def test_get_reads_from_the_job_store() -> None:
    node = Node(name="train")
    store = Mock()
    store.get_record.return_value = {"loss": 1.0}
    job = SimpleNamespace(store=store)

    assert LoggingJob.get(job, node) == {"loss": 1.0}
    store.get_record.assert_called_once_with(node)


def test_logging_job_does_not_require_checkpointing() -> None:
    assert issubclass(LoggingJob, BasicJob)
    assert not issubclass(LoggingJob, CheckpointedJob)
    assert LoggingJob.state_init is BasicJob.state_init
    assert LoggingJob.state_restore is BasicJob.state_restore
    assert LoggingJob.STORE is RecordStore


def test_record_marks_current_node_after_queueing() -> None:
    node = Node(name="train")
    store = Mock()
    payload = {"attention": np.ones(2)}
    job = SimpleNamespace(
        node=node,
        store=store,
        _node_written=False,
        main_process=Mock(return_value=True),
        _normalize_log=LoggingJob._normalize_log,
    )

    LoggingJob.artifact(job, "validation", {"step": 1}, payload)

    store.artifact.assert_called_once_with(node, "validation", {"step": 1}, payload)
    assert job._node_written is True


def test_save_marks_checkpoint_metadata_without_mutating_input(tmp_path):
    node = Node(name="train")
    metadata = {"loss": 1.25}
    store = Mock()
    store.blob.return_value = nullcontext(tmp_path)
    manager = Mock()
    job = SimpleNamespace(
        node=node,
        store=store,
        _node_written=False,
        chkpt_manager=manager,
        key=jax.random.PRNGKey(0),
        main_process=Mock(return_value=False),
    )
    state = {"params": np.arange(4)}

    CheckpointedJob.save(job, state, metadata)

    store.blob.assert_called_once_with(
        node,
        {"loss": 1.25, "_x_checkpoint": True},
    )
    manager.save.assert_called_once_with(state, tmp_path / "checkpoint")
    assert metadata == {"loss": 1.25}
    assert job._node_written is True


def test_restoreable_save_omits_an_inherited_job_name(monkeypatch) -> None:
    from theseus.training.base import BaseTrainer

    checkpoint_save = Mock()
    monkeypatch.setattr(RestoreableJob, "JOB_NAME", "inherited", raising=False)
    monkeypatch.setattr(CheckpointedJob, "save", checkpoint_save)
    job = object.__new__(BaseTrainer)
    state = object()
    metadata = {"loss": 1.25}

    RestoreableJob.save(job, state, metadata)

    checkpoint_save.assert_called_once_with(state, {"loss": 1.25})
    assert metadata == {"loss": 1.25}


@pytest.mark.parametrize("job_key", [None, "external/not-imported"])
def test_from_node_rejects_unknown_job(tmp_path, job_key):
    spec = ExecutionSpec.local(str(tmp_path))
    node = Node(name="external", nonce="abcdef")
    store = ObjectStore(spec.hardware)
    try:
        metadata = {"_x_checkpoint": True}
        if job_key is not None:
            metadata["_x_job"] = job_key
        with store.blob(node, metadata) as blob:
            (blob / "config.yaml").write_text("{}")
    finally:
        store.close()

    with pytest.raises(
        ValueError, match="Unknown checkpoint job.*Import the job's module"
    ):
        RestoreableJob.from_node(node, spec)


def test_state_restore_restores_randomness_checkpoint_and_applies(
    tmp_path, monkeypatch
):
    blob = tmp_path / "blob"
    blob.mkdir()
    randomness = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "jax_random": 17,
    }
    np.save(blob / "rng.npy", randomness)

    restored = object()
    manager = Mock()
    manager.restore.return_value = restored, None
    apply = Mock()
    python_setstate = Mock()
    numpy_set_state = Mock()
    monkeypatch.setattr(random, "setstate", python_setstate)
    monkeypatch.setattr(np.random, "set_state", numpy_set_state)
    job = SimpleNamespace(
        base=Node(name="source"),
        chkpt_manager=manager,
        template="template",
        surgery=Mock(),
        apply=apply,
        key=None,
    )

    metadata = {"blob": blob, "loss": 1.25}
    CheckpointedJob.state_restore(job, metadata)

    manager.restore.assert_called_once_with(
        blob / "checkpoint", "template", allow_partial=True
    )
    job.surgery.assert_not_called()
    python_setstate.assert_called_once_with(randomness["python_random"])
    numpy_state = numpy_set_state.call_args.args[0]
    assert numpy_state[0] == randomness["numpy_random"][0]
    np.testing.assert_array_equal(numpy_state[1], randomness["numpy_random"][1])
    assert numpy_state[2:] == randomness["numpy_random"][2:]
    np.testing.assert_array_equal(job.key, jax.random.PRNGKey(17))
    apply.assert_called_once_with(restored, metadata)


def test_state_restore_uses_default_key_without_randomness(tmp_path):
    manager = Mock()
    restored = object()
    manager.restore.return_value = restored, None
    apply = Mock()
    job = SimpleNamespace(
        base=None,
        chkpt_manager=manager,
        template="template",
        surgery=Mock(),
        apply=apply,
        key=None,
    )

    metadata = {"blob": tmp_path}
    CheckpointedJob.state_restore(job, metadata)

    np.testing.assert_array_equal(job.key, jax.random.PRNGKey(0))
    manager.restore.assert_called_once_with(
        tmp_path / "checkpoint", "template", allow_partial=True
    )
    job.surgery.assert_not_called()
    apply.assert_called_once_with(restored, metadata)


def test_state_restore_initializes_only_surgery_leaves(tmp_path):
    manager = Mock()
    restored = {
        "existing": np.asarray(1),
        "new": jax.ShapeDtypeStruct((), np.dtype("int64")),
    }
    partial = {"existing": False, "new": True}
    manager.restore.return_value = restored, partial
    surgery = Mock(return_value={"existing": None, "new": np.asarray(2)})
    apply = Mock()
    job = SimpleNamespace(
        base=None,
        chkpt_manager=manager,
        template="template",
        surgery=surgery,
        apply=apply,
        key=None,
    )
    metadata = {"blob": tmp_path}

    CheckpointedJob.state_restore(job, metadata)

    surgery.assert_called_once_with(partial)
    applied, applied_metadata = apply.call_args.args
    assert applied_metadata == metadata
    assert applied["existing"] == 1
    assert applied["new"] == 2


def test_state_restore_rejects_node_without_blob():
    node = Node(name="source")
    job = SimpleNamespace(base=node)

    with pytest.raises(
        ValueError,
        match=f"No restorable object is associated with node={node.serialize()}",
    ):
        CheckpointedJob.state_restore(job, {})


@pytest.mark.parametrize(
    "has_base,resume,required",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_setup_prepares_state_without_collectives_or_execution(
    monkeypatch, has_base, resume, required
):
    barrier = Mock()
    monkeypatch.setattr("theseus.job.multihost_utils.sync_global_devices", barrier)
    base = Node(name="source", nonce="abcdef", seq=3) if has_base else None
    store = Mock()
    metadata = {"blob": "/checkpoint"}
    store.query.return_value.node.return_value.select.return_value = [metadata]
    job = SimpleNamespace(
        base=base,
        node=Node(name="target"),
        store=store,
        spec=SimpleNamespace(name="target"),
        _resume_required=required,
        _setup_complete=False,
        node_name=Mock(return_value="target"),
        state_restore=Mock(),
        state_init=Mock(),
        state_reset=Mock(),
        _synchronize_node=Mock(),
        run=Mock(),
        finish=Mock(),
    )
    BasicJob.setup(job, resume=resume)
    prepared_node = job.node
    BasicJob.setup(job, resume=not resume)
    assert job.node is prepared_node
    assert job._setup_complete
    if has_base:
        job.state_restore.assert_called_once_with(metadata)
        job.state_init.assert_not_called()
        if resume or required:
            assert job.node == base
        else:
            assert job.node.parent == base.serialize()
    else:
        job.state_init.assert_called_once_with()
        job.state_restore.assert_not_called()
        assert job.node.name == "target"
    assert job.state_reset.call_count == (0 if resume or required else 1)
    barrier.assert_not_called()
    job._synchronize_node.assert_not_called()
    store.value.assert_not_called()
    job.run.assert_not_called()
    job.finish.assert_not_called()


def test_setup_can_retry_after_failed_initialization():
    job = SimpleNamespace(
        _setup_complete=False,
        _resume_required=False,
        base=None,
        spec=SimpleNamespace(name="retry"),
        state_init=Mock(side_effect=[ValueError("initialization failed"), None]),
        state_reset=Mock(),
        node_name=Mock(return_value="retry"),
    )
    with pytest.raises(ValueError, match="initialization failed"):
        BasicJob.setup(job)
    assert not job._setup_complete
    BasicJob.setup(job)
    BasicJob.setup(job)
    assert job._setup_complete
    assert job.state_init.call_count == 2
    job.state_reset.assert_called_once_with()
