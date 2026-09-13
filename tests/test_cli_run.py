from datetime import datetime
from functools import partial
import pytest
from theseus.base import ExecutionSpec, Node
from theseus.cli.interface.data import RunData, WorkspaceData
from theseus.cli.interface.state import InterfaceState
from theseus.cli.interface.plot_math import (
    format_tick,
    format_timestamp,
    format_timestamp_tick,
    interpolate,
    ticks,
    twema,
)
from theseus.store import ObjectStore


@pytest.fixture(autouse=True)
def isolate_ui_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def make_run(tmp_path, steps: int = 12) -> tuple[str, str]:
    name, nonce = "alpha.group.train", "abcdef"
    store = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    for seq in range(steps):
        values: dict[str, float | int | str] = {}
        if seq % 2 == 0:
            values = {
                "step": seq,
                "train/loss": float(steps - seq),
                "tokens": seq * 8,
            }
            if seq == 0:
                values.update({f"detail/{index}": str(index) for index in range(20)})
        if seq in {3, 9}:
            values["_x_checkpoint"] = True
        store.value(Node(name=name, nonce=nonce, seq=seq), values)
    store.close()
    return name, nonce


def test_run_data_bounds_timeline_and_plot_series(tmp_path) -> None:
    name, nonce = make_run(tmp_path, steps=100)
    other_name, other_nonce = "beta.group.eval", "fedcba"
    writer = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    for seq in range(100):
        writer.value(
            Node(name=other_name, nonce=other_nonce, seq=seq),
            {"step": seq, "train/loss": float(100 - seq)},
        )
    writer.close()
    store = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    data = RunData(store, name, nonce)
    workspace = WorkspaceData((data, RunData(store, other_name, other_nonce)))

    timeline = data.timeline(12)

    assert len(timeline.bins) == 12
    assert timeline.first == 0
    assert timeline.last == 99
    assert any(marker & 2 for marker in timeline.bins)
    assert data.nearest(53) == 53
    assert data.scalar_keys == ("step", "timestamp", "tokens", "train/loss")
    assert len(data.series("step", "train/loss", 8)) <= 9
    assert data.plot_series("step", "train/loss", 8)[0].checkpoints == (
        (3.0, 97.0),
        (9.0, 91.0),
    )
    assert workspace.scalar_keys == ("step", "timestamp", "train/loss")
    assert len(workspace.plot_series("step", "train/loss", 8)) == 2
    assert len(workspace.plot_series("step", "tokens", 8)) == 1
    frame = data.to_dataframe()
    assert len(frame) == 100
    assert set(frame["_x_name"]) == {name}
    assert set(frame["_x_nonce"]) == {nonce}
    assert frame.loc[frame["_x_seq"] == 3, "_x_checkpoint"].item()
    assert str(frame["timestamp"].dtype).startswith("datetime64")
    stored_timestamps = {
        node.seq: values["_x_timestamp"]
        for node, values in (
            store.query()
            .name(name)
            .nonce(nonce)
            .select(return_nodes=True, raw=True, keys=("_x_timestamp",))
        )
    }
    assert all(
        values["timestamp"] == stored_timestamps[seq] / 1_000_000_000
        for seq, (_, values) in data.rows.items()
    )
    workspace_frame = workspace.to_dataframe()
    assert len(workspace_frame) == 200
    assert set(workspace_frame["_x_nonce"]) == {nonce, other_nonce}
    assert str(workspace_frame["timestamp"].dtype).startswith("datetime64")
    store.close()


def test_interface_state_persists_views_and_plot_configuration(tmp_path) -> None:
    path = tmp_path / "state" / "ui.sqlite3"
    state = InterfaceState(path)
    view = state.add_view(tmp_path, "alpha.group.train", "abcdef")
    workspace_view = state.add_view(tmp_path)
    plot = state.add_plot(view)
    state.update_plot(
        plot,
        x="step",
        y="train/loss",
        log_x=False,
        log_y=True,
        checkpoints=False,
        smoothing=0.4,
    )
    state.close()

    restored = InterfaceState(path)

    assert restored.views(tmp_path, "alpha.group.train", "abcdef") == [view]
    assert restored.views(tmp_path) == [workspace_view]
    saved = restored.plots(view)[0]
    assert (
        saved.x,
        saved.y,
        saved.log_x,
        saved.log_y,
        saved.checkpoints,
        saved.smoothing,
    ) == (
        "step",
        "train/loss",
        False,
        True,
        False,
        0.4,
    )
    restored.delete_view(view)
    assert restored.views(tmp_path, "alpha.group.train", "abcdef") == []
    assert restored.plots(view) == []
    restored.close()


def test_chart_twema_ticks_and_axis_keys(tmp_path) -> None:
    points = tuple((float(index), float(index)) for index in range(101))

    assert twema(points, 0) == points
    smoothed = twema(points, 1)
    assert smoothed[1][1] == pytest.approx(1 / 1.999)
    assert format_tick(5_250) == "5.25K"
    assert format_tick(5_500_000) == "5.5M"
    assert format_tick(-5_000_000_000) == "-5B"
    assert format_tick(5_000_000_000_000) == "5T"
    assert format_tick(0.025) == "0.025"
    assert len(ticks(3, 8, False, 5)) == 5
    timestamp = datetime(2026, 9, 4, 12, 34, 56, 789000).timestamp()
    assert format_timestamp(timestamp) == "2026-09-04 12:34:56.789"
    assert format_timestamp_tick(timestamp, 1) == "12:34:56.789"
    timestamp_ticks = ticks(
        timestamp,
        timestamp + 30,
        False,
        3,
        partial(format_timestamp_tick, span=30),
    )
    assert [label for _, label in timestamp_ticks] == [
        "12:34:56",
        "12:35:11",
        "12:35:26",
    ]
    assert interpolate([(0, 2), (10, 6)], 5) == 4

    name, nonce = make_run(tmp_path)
    store = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    data = RunData(store, name, nonce)
    keys = tuple(dict.fromkeys(("step", *data.scalar_keys)))
    assert keys.count("step") == 1
    assert keys.count("timestamp") == 1
    store.close()


def test_run_data_retains_execution_ids_for_logs(tmp_path):
    spec = ExecutionSpec.local(str(tmp_path), name="ui")
    data_store = ObjectStore(spec.hardware, execution_id="dispatch1")
    node = Node(name="alpha.group.train", nonce="abcdef")
    data_store.value(node, {"loss": 1.0})
    data_store.close()
    data = RunData(data_store, node.name, node.nonce)
    assert data.executions == ("dispatch1",)
    assert "_x_execution" not in data.scalar_keys
