"""Public object/record/query semantics against persistent ClickHouse tables."""
from pathlib import Path
from shutil import copyfile, copytree
from typing import Any, cast

import jax
import numpy as np
import pytest

import theseus.store as store_module
from theseus.base import Node
from theseus.base.hardware import HardwareResult, local
from theseus.store import ObjectReader, ObjectStore, RecordStore


@pytest.fixture(autouse=True)
def metadata_home(tmp_path, monkeypatch):
    # Keep tests out of the user's real metadata directory and close every handle.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    readers = []
    initialize = ObjectReader.__init__

    def track(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        readers.append(self)

    monkeypatch.setattr(ObjectReader, "__init__", track)
    yield
    for reader in reversed(readers):
        try:
            reader.close()
        except RuntimeError:
            pass


@pytest.fixture
def hardware(tmp_path: Path) -> HardwareResult:
    return local(str(tmp_path), "-")


def test_local_constructs_store_without_an_execution_spec(tmp_path: Path) -> None:
    store = ObjectStore.local(str(tmp_path))

    assert store.root() == tmp_path / "objects"
    store.close()


def test_wide_selection_preserves_sparse_merge_and_query_semantics(
    hardware, monkeypatch
):
    # Even listing these identifiers once exceeds the default SQL parser limit.
    metrics = {f"train/{'long_metric_' * 16}{i}{{}}": float(i) for i in range(1500)}
    store = ObjectStore(hardware)
    first = Node(name="wide", nonce="abcdef", seq=1)
    second = Node(name="wide", nonce="abcdef", seq=2)
    store.value(first, metrics)
    store.close()
    store = ObjectStore(hardware)  # Keep parts with different schemas separate.
    store.value(first, {"loss": 2.0, "label`\\雪": "kept"})
    store.value(first, {"loss": 1.0})
    store.value(second, {"loss": 3.0})
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    store.value(first, {"loss": 99.0, "worker_only": 7})
    store.close()
    store.start()

    rows = store.query().sort("loss", ascending=False).select(return_nodes=True)
    assert rows == [
        (second, {"loss": 3.0}),
        (first, {**metrics, "loss": 1.0, "label`\\雪": "kept", "worker_only": 7}),
    ]
    assert store.query().where("loss", "<", 2).select(keys=["loss"]) == [{"loss": 1.0}]
    raw = store.query().node(first).select(raw=True)[0]
    assert raw["_x_seq"] == 1
    assert raw["loss"] == 1.0
    assert all(raw[key] == value for key, value in metrics.items())
    assert store.query().latest().select(keys=["loss"]) == [{"loss": 1.0}]


def test_artifact_preserves_existing_node_specification(hardware: HardwareResult) -> None:
    store = RecordStore(hardware)
    node = Node(name="checkpoint", nonce="abcdef", seq=3).next()
    with store.blob(node, {"_x_checkpoint": True}) as directory:
        pass
    path = directory / "nodespec.json"
    original = node.model_dump_json(indent=2).encode()
    path.write_bytes(original)
    # Time-travel queries identify a node without carrying its parent field.
    store.artifact(Node.deserialize(node.serialize()), "probe", {}, {"value": 1})
    store.close()
    store.start()
    assert path.read_bytes() == original
    assert store.parents(node) == [Node.deserialize(node.parent)]


def test_record_store_writes_marks_and_decodes_records(
    hardware: HardwareResult,
) -> None:
    store = RecordStore(hardware)
    node = Node(name="analysis", nonce="abcdef")
    payload = {"attention": 2}

    store.artifact(node, "validation", {"step": 1}, payload)
    store.close()
    store.start()

    raw = store.query().node(node).select(raw=True)[0]
    blob = raw["blob"]
    assert isinstance(blob, Path)
    assert raw["_x_record"] is True
    expected = {"step": 1, "blob": blob, "payload": [payload]}
    assert store.get_record(node) == expected
    assert store.query().artifact().select() == [expected]
    assert store.query().artifact().select(return_nodes=True) == [(node, expected)]


def test_non_scalar_values_are_rejected_before_the_writer_is_poisoned(
    hardware: HardwareResult,
) -> None:
    store = ObjectStore(hardware)
    node = Node(name="metadata")

    with pytest.raises(TypeError, match="numpy.ndarray.*numpy_random"):
        store.value(node, {"numpy_random": cast(Any, np.arange(4))})
    with pytest.raises(TypeError, match="tuple.*python_random"):
        with store.blob(node, {"python_random": cast(Any, (1, (2, 3), None))}):
            pytest.fail("invalid metadata must be rejected before yielding")

    assert store.error is None
    store.value(node, {"loss": cast(Any, np.float32(1.25))})
    store.close()
    store.start()
    assert store.query().node(node).select() == [{"loss": 1.25}]


def test_nodes_and_parents_are_distinct_exact_and_sorted(
    hardware: HardwareResult,
) -> None:
    store = ObjectStore(hardware)
    first_parent = Node(name="parent-a", nonce="aaaaaa", seq=1)
    second_parent = Node(name="parent-b", nonce="bbbbbb", seq=2)
    child = Node(
        name="child' OR 1 = 1 --",
        nonce="abc'def",
        seq=3,
        parent=second_parent.serialize(),
    )
    alternate = child.model_copy(update={"parent": first_parent.serialize()})
    unparented = Node(name="solo", nonce="cccccc", seq=0)

    store.value(child, {"child": 1})
    store.value(alternate, {"alternate": 2})
    store.value(unparented, {"solo": 3})
    store.value(unparented, {"duplicate": 4})
    store.close()
    store.start()

    assert store.query().all() == [
        Node(name=child.name, nonce=child.nonce, seq=child.seq),
        unparented,
    ]
    expected_parents = sorted(
        [first_parent, second_parent], key=lambda parent: parent.serialize()
    )
    assert store.parents(child) == expected_parents
    assert store.parents(unparented) == []
    assert store.parents(Node(name="missing")) == []


def test_object_reader_shares_store_queries_without_starting_a_writer(
    tmp_path: Path,
) -> None:
    store = ObjectStore.local(tmp_path)
    parent = Node(name="parent")
    child = Node(name="child", parent=parent.serialize())
    store.value(child, {"loss": 1.0})
    store.close()

    reader = ObjectReader.local(tmp_path)
    reader.start()

    assert isinstance(store, ObjectReader)
    assert reader.query().node(child).select() == [{"loss": 1.0}]
    assert reader.parents(child) == [parent]


def test_query_builder_folds_filters_and_sorts_nodes(
    hardware: HardwareResult,
) -> None:
    store = ObjectStore(hardware)
    nodes = [
        Node(name="alpha.shared.run", nonce="bbbbbb", seq=0),
        Node(name="alpha.shared.run", nonce="aaaaaa", seq=2),
        Node(name="beta.shared.run", nonce="aaaaaa", seq=1),
        Node(name="alpha.other.alt", nonce="cccccc", seq=0),
    ]
    for node, score in zip(nodes, (0.7, 0.9, 0.8, 0.95), strict=True):
        store.value(node, {"logs/loss": 1.0, "name": "logged"})
        store.value(node, {"eval/score": score})
    missing = Node(name="gamma.other.missing", nonce="dddddd", seq=0)
    store.value(missing, {"logs/loss": 1.0})
    store.value(nodes[3], {"eval/score": 0.1})
    store.value(nodes[0], {"_x_record": True})
    store.value(nodes[1], {"_x_record": True})
    store.value(nodes[1], {"_x_checkpoint": True, "_x_job": "tests/train"})
    store.value(nodes[2], {"_x_checkpoint": True, "_x_job": "tests/evaluate"})
    store.close()
    store.start()

    assert store.query().has("logs/loss").where("eval/score", ">", 0.5).all() == [
        nodes[1],
        nodes[0],
        nodes[2],
    ]
    assert store.query().checkpoint().all() == [nodes[1], nodes[2]]
    assert store.query().artifact().all() == [nodes[1], nodes[0]]
    assert store.query().checkpoint().artifact().all() == [nodes[1]]
    assert store.query().name(nodes[0].name).nonce(nodes[0].nonce).all() == [nodes[0]]
    assert store.query().name(nodes[1].name).nonce(nodes[1].nonce).seq(
        nodes[1].seq
    ).all() == [nodes[1]]
    assert store.query().node(nodes[1].serialize()).all() == [nodes[1]]
    assert store.query().spec("alpha", "shared", "run").select(return_nodes=True) == [
        (nodes[1], {"eval/score": 0.9, "logs/loss": 1.0, "name": "logged"}),
        (nodes[0], {"eval/score": 0.7, "logs/loss": 1.0, "name": "logged"}),
    ]
    assert store.query().job("tests/train").has("logs/loss").all() == [nodes[1]]
    assert store.query().job("missing").all() == []
    assert store.query().spec(project="alpha").all() == [
        nodes[3],
        nodes[1],
        nodes[0],
    ]
    assert store.query().spec(group="shared", run="run").all() == [
        nodes[1],
        nodes[0],
        nodes[2],
    ]
    assert store.query().spec("alpha", "shared", "run").all() == [
        nodes[1],
        nodes[0],
    ]
    assert store.query().sort("eval/score", ascending=False).all() == [
        nodes[1],
        nodes[2],
        nodes[0],
        nodes[3],
        missing,
    ]
    assert store.query().sort("eval/score").all() == [
        nodes[3],
        nodes[0],
        nodes[2],
        nodes[1],
        missing,
    ]
    assert store.query().sort("missing").all() == [
        nodes[3],
        nodes[1],
        nodes[0],
        nodes[2],
        missing,
    ]
    builder = store.query().node(nodes[0].serialize())
    assert builder.all() == [nodes[0]]
    assert builder.all() == [nodes[3], nodes[1], nodes[0], nodes[2], missing]

    builder.node(nodes[1].serialize())
    assert builder.select(return_nodes=True) == [
        (nodes[1], {"eval/score": 0.9, "logs/loss": 1.0, "name": "logged"})
    ]
    raw = (
        store.query()
        .node(nodes[1].serialize())
        .select(return_nodes=True, raw=True, keys=["_x_write"])
    )
    assert raw[0][0] == nodes[1]
    assert set(raw[0][1]) == {"_x_name", "_x_nonce", "_x_seq", "_x_write"}
    assert builder.all() == [nodes[3], nodes[1], nodes[0], nodes[2], missing]
    assert store.query().has("missing").all() == []
    with pytest.raises(ValueError, match="reserved fields"):
        store.query().has("_x_job")
    assert store.query().sort("_x_seq").all() == [
        nodes[3],
        nodes[0],
        missing,
        nodes[2],
        nodes[1],
    ]
    with pytest.raises(ValueError, match="raw=True"):
        store.query().select(keys=["_x_write"])
    with pytest.raises(ValueError, match="unsupported query operator"):
        store.query().where("eval/score", "LIKE", 0.5)


@pytest.mark.parametrize("operation", ["finished", "parents", "select"])
def test_query_size_limit_applies_to_all_reader_paths(
    hardware: HardwareResult, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = ObjectStore(hardware, "large-query", "train")
    parent = Node(name="parent", nonce="aaaaaa")
    node = Node(name="child", nonce="bbbbbb", parent=parent.serialize())
    store.value(node, {"_x_finished": True, "score": 1.0})
    store.close()
    store.start()
    source, columns = store._snapshot()
    # Exercise the parser limit without creating thousands of physical files.
    source = f"(SELECT * FROM {source} WHERE notEmpty('{'x' * 300000}'))"
    monkeypatch.setattr(store, "_snapshot", lambda: (source, columns))
    if operation == "finished":
        assert store.query().execution("large-query").tag("train").finished().all() == [
            node.model_copy(update={"parent": None})
        ]
    elif operation == "parents":
        assert store.parents(node) == [parent]
    else:
        assert store.query().node(node).select(keys=["score"]) == [{"score": 1.0}]


def test_query_discovers_tagged_execution_state(hardware: HardwareResult) -> None:
    complete = ObjectStore(hardware, "execution-a", "pretrain")
    first = Node(name="tests.chain.run", nonce="aaaaaa", seq=1)
    terminal = first.next()
    complete.value(first, {"_x_checkpoint": True})
    complete.value(terminal, {"_x_finished": True})
    complete.close()

    active = ObjectStore(hardware, "execution-a", "midtrain")
    latest = Node(name="tests.chain.run", nonce="bbbbbb", seq=4)
    active.value(latest, {"_x_checkpoint": True})
    active.close()

    unrelated = ObjectStore(hardware, "execution-b", "pretrain")
    unrelated.value(Node(name="tests.chain.run", nonce="cccccc"), {})
    unrelated.close()
    complete.start()

    assert complete.query().execution("execution-a").tag(
        "pretrain"
    ).finished().latest().all() == [terminal.model_copy(update={"parent": None})]
    assert (
        complete.query()
        .execution("execution-a")
        .tag("pretrain")
        .finished(False)
        .latest()
        .all()
        == []
    )
    assert complete.query().execution("execution-a").tag("midtrain").finished(
        False
    ).latest().all() == [latest]
    assert (
        complete.query().execution("execution-a").tag("midtrain").finished().all() == []
    )

    raw = complete.query().node(first).select(raw=True)[0]
    assert raw["_x_execution"] == "execution-a"
    assert raw["_x_tag"] == "pretrain"
    assert (
        complete.query()
        .execution("execution-a")
        .select(raw=True, keys=["_x_execution", "_x_tag"])
    )


def test_query_folds_chunks_repaths_blobs_and_matches_serialized_node(
    hardware: HardwareResult, tmp_path: Path
) -> None:
    store = ObjectStore(hardware)
    node = Node(name="train:branch", nonce="abcdef", seq=3)
    store.value(
        node,
        {"logs/loss": 2.0, "shared": 1.0, "_x_job": "tests/train"},
    )
    store.close()
    store.start()

    store.start()
    store.value(node, {"eval/score": 0.75, "shared": 2.0})
    next_node = node.next()
    store.value(next_node, {"future": 4.0})
    with store.blob(node) as blob:
        (blob / "state").write_text("checkpoint")
    store.close()
    store.start()

    assert Node.model_validate_json((blob / "nodespec.json").read_text()) == node
    expected = {
        "blob": blob,
        "eval/score": 0.75,
        "logs/loss": 2.0,
        "shared": 2.0,
    }
    assert store.query().node(node).select() == [expected]
    assert store.query().node(node.serialize()).select() == [expected]
    assert store.query().node(node).select(return_nodes=True) == [(node, expected)]
    raw = store.query().node(node).select(raw=True)[0]
    assert raw["_x_job"] == "tests/train"
    assert raw["_x_name"] == node.name
    assert raw["blob"] == blob
    assert (
        store.query().node(Node(name=node.name, nonce=node.nonce, seq=2)).select() == []
    )
    assert store.parents(next_node) == [node]

    store.close()
    relocated_root = tmp_path / "relocated"
    copytree(store.root(), relocated_root / "objects")
    copyfile(tmp_path / ".id", relocated_root / ".id")
    # Metadata location is explicit and independent of the blob root.
    relocated = ObjectReader.local(relocated_root, metadata=store._metadata)
    relocated.start()
    expected["blob"] = relocated.root() / "blobs" / node.serialize()
    assert relocated.query().node(node).select() == [expected]



def test_blob_failure_is_not_finalized(hardware: HardwareResult) -> None:
    store = ObjectStore(hardware)
    node = Node(name="partial")

    with pytest.raises(LookupError, match="producer failed"):
        with store.blob(node, {"kind": "checkpoint"}) as blob:
            (blob / "partial").write_text("partial")
            raise LookupError("producer failed")
    store.close()
    store.start()

    assert blob.exists()
    assert not (blob / "nodespec.json").exists()
    assert store.query().node(node).select() == []
    assert store.query().all() == []
    assert store.parents(node) == []


def test_non_primary_blob_records_metadata_without_nodespec(
    hardware: HardwareResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = hardware.hosts[0]
    multihost_hardware = hardware.model_copy(update={"hosts": [host, host]})
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    store = ObjectStore(multihost_hardware)
    node = Node(name="shard")

    with store.blob(node, {"kind": "checkpoint"}) as blob:
        (blob / "shard-1").write_text("data")
    store.close()
    store.start()

    assert not (blob / "nodespec.json").exists()
    assert store.query().node(node).select() == [{"blob": blob, "kind": "checkpoint"}]


def test_live_node_ticks_do_not_relabel_queued_writes(hardware, monkeypatch):
    monkeypatch.setattr(RecordStore, "_start_records", lambda self: None)
    store = RecordStore(hardware)
    node = Node(name="live", nonce="abcdef")
    original = node.model_copy()
    # Hold the record worker until after ticking, making the race deterministic.
    from queue import Queue

    store._record_queue = Queue()
    store.value(node, {"loss": 1.0})
    store.artifact(node, "attention", {}, {"value": 3})
    node.update(node.next())
    queued = store._record_queue.get_nowait()
    assert queued[0] == original
    assert queued[0] is not node
    store.close()
    store.start()
    assert store.query().node(original).select()[0]["loss"] == 1.0
    assert store.query().node(node).select() == []
