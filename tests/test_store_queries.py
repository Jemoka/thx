"""Candidate pruning must preserve folded values and execution scope."""
from unittest.mock import Mock

import pytest

from theseus.base import Node
from theseus.store import ObjectReader, ObjectStore
from tests.test_store import metadata_home
from tests.test_store_clickhouse import clickhouse_server


@pytest.fixture(params=["embedded", "server"])
def query_store(tmp_path, request):
    metadata = str(tmp_path / "query.chdb") if request.param == "embedded" else request.getfixturevalue("clickhouse_server")
    store = ObjectStore.local(tmp_path, metadata=metadata)
    try:
        yield store
    finally:
        store.close()


@pytest.mark.parametrize("marker", ["checkpoint", "artifact", "job", "predicate"])
def test_candidates_keep_sparse_events_and_reject_losing_values(query_store, marker):
    store = query_store
    first = Node(name="project.group.first", nonce="aaaaaa", seq=1)
    second = Node(name="project.group.second", nonce="bbbbbb", seq=2)
    third = Node(name="project.group.third", nonce="cccccc", seq=3)
    # Explicit ordering makes the latest candidate independent of append order.
    store._write_values([
        {"_x_name": node.name, "_x_nonce": node.nonce, "_x_seq": node.seq,
         "_x_prc_idx": prc, "_x_write": write, "_x_event": str(write), **values}
        for node, prc, write, values in [
            (first, 0, 1, {"tokens": 10, "metric": 1.5}),
            (first, 0, 2, {"_x_checkpoint": True, "_x_record": True, "_x_job": "train"}),
            (first, 1, 100, {"tokens": 1, "worker_metric": 8}),
            (second, 0, 3, {"tokens": 1, "metric": 2.5}),
            (second, 0, 4, {"_x_checkpoint": True, "_x_record": True, "_x_job": "train"}),
            (second, 1, 200, {"tokens": 100}),
            (third, 0, 5, {"tokens": 20, "metric": 3.5}),
            (third, 0, 6, {"_x_checkpoint": True, "_x_record": True, "_x_job": "train"}),
        ]
    ])
    query = store.query().spec(project="project").where("tokens", ">=", 10)
    if marker == "checkpoint":
        query.checkpoint()
    elif marker == "artifact":
        query.artifact()
    elif marker == "job":
        query.job("train")
    # The latest qualifying node is first: process 1 contributes write time but
    # cannot override process 0's token count. The other candidate must fail HAVING.
    assert query.latest().select(return_nodes=True) == [
        (first, {"tokens": 10, "metric": 1.5, "worker_metric": 8})
    ]
    assert store.query().where("tokens", "<", 10).select(return_nodes=True) == [
        (second, {"tokens": 1, "metric": 2.5})
    ]
    assert store.query().has("worker_metric").select(keys=["tokens"]) == [{"tokens": 10}]


def test_latest_retains_scope_in_both_reads_and_finished_is_scope_wide(query_store, monkeypatch):
    store = query_store
    node = Node(name="project.group.run", nonce="aaaaaa", seq=1)
    terminal = node.next()
    for execution, tag, write, fields in [
        ("active", "train", 1, {"tokens": 10, "metric": 2.0}),
        ("active", "train", 2, {"_x_checkpoint": True}),
        ("other", "train", 100, {"tokens": 100, "metric": 99.0}),
        ("active", "eval", 101, {"tokens": 100, "metric": 98.0}),
    ]:
        store._write_values([{
            "_x_name": node.name, "_x_nonce": node.nonce, "_x_seq": node.seq,
            "_x_prc_idx": 0, "_x_write": write, "_x_event": str(write),
            "_x_execution": execution, "_x_tag": tag, **fields,
        }])
    store.execution_id = "active"
    store.tag = "train"
    store.value(terminal, {"_x_finished": True})
    store.close()
    store.start()
    requests = Mock(wraps=store._query_arrow)
    monkeypatch.setattr(store, "_query_arrow", requests)
    assert store.query().execution("active").tag("train").finished().checkpoint().where("tokens", ">=", 10).latest().select() == [{"tokens": 10, "metric": 2.0}]
    # One fresh schema read and one SQL statement, including the completion check.
    assert requests.call_count == 2
    assert store.query().execution("active").tag("train").finished(False).latest().all() == []
    assert store.query().execution("active").tag("train").where("tokens", ">", 20).latest().select() == []


def test_identity_sorting_and_literal_project_prefixes(query_store):
    store = query_store
    nodes = [
        Node(name="a+b.group.run", nonce="bbbbbb", seq=2),
        Node(name="a+b.group.run", nonce="aaaaaa", seq=1),
        Node(name="a+b.other.run", nonce="cccccc", seq=3),
        Node(name="aab.group.run", nonce="dddddd", seq=0),
        Node(name="a+bextra.group.run", nonce="eeeeee", seq=0),
        Node(name="a+b.grouping.run", nonce="ffffff", seq=4),
    ]
    for node in nodes:
        store.value(node, {"metric": node.seq})
    store.close()
    store.start()
    assert store.query().spec(project="a+b", group="group").sort("_x_seq").all() == [nodes[1], nodes[0]]
    assert store.query().spec(project="a+b", group="group", run="run").sort("_x_nonce", ascending=False).all() == [nodes[0], nodes[1]]
    assert store.query().spec(group="other", run="run").all() == [nodes[2]]
    assert store.query().spec(project="missing").latest().select() == []


def test_schema_reads_are_fresh_and_missing_tables_are_empty(tmp_path, monkeypatch):
    store = ObjectStore.local(tmp_path)
    store.close()
    reader = ObjectReader.local(tmp_path)
    try:
        reader._client.command(f"DROP TABLE {reader._table}")
        requests = Mock(wraps=reader._query_arrow)
        monkeypatch.setattr(reader, "_query_arrow", requests)
        assert reader.query().all() == []
        assert requests.call_count == 1
    finally:
        reader.close()
    store.start()
    store.value(Node(name="new"), {"new_column": 7})
    store.close()
    reader.start()
    try:
        assert reader.query().select() == [{"new_column": 7}]
    finally:
        reader.close()


def test_latest_without_predicates_preserves_ties_and_sparse_values(query_store):
    store = query_store
    nodes = [
        Node(name="project.group.b", nonce="aaaaaa", seq=4),
        Node(name="project.group.a", nonce="bbbbbb", seq=4),
        Node(name="project.group.a", nonce="aaaaaa", seq=4),
        Node(name="project.group.z", nonce="cccccc", seq=3),
    ]
    store._write_values([
        {"_x_name": node.name, "_x_nonce": node.nonce, "_x_seq": node.seq,
         "_x_prc_idx": 0, "_x_write": write, "_x_event": str(write), **values}
        for node in nodes
        for write, values in [(1, {"metric": node.seq}), (10, {"extra": 7})]
    ])
    winner = nodes[2]
    assert store.query().latest().all() == [winner]
    assert store.query().latest().select(return_nodes=True) == [
        (winner, {"metric": 4, "extra": 7})
    ]
    assert store.query().latest().select(keys=[]) == [{}]
    assert store.query().latest().select(keys=["metric"]) == [{"metric": 4}]
    row = store.query().latest().select(raw=True)[0]
    assert row["_x_name"] == winner.name
    assert row["_x_nonce"] == winner.nonce
    assert row["metric"] == 4
