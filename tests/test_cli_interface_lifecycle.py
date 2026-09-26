"""Client deletion must stop callbacks before they reuse SQLite or the UI."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from theseus.cli.interface import driver
from theseus.cli.interface.location import Location
from theseus.cli.interface.state import InterfaceState


@pytest.fixture
def interface(tmp_path):
    app = driver.TheseusInterface.__new__(driver.TheseusInterface)
    app.closed = False
    app.restoring = False
    app.initial_location = Location("")
    app.root_path = tmp_path
    app.state = InterfaceState(tmp_path / "ui.sqlite3")
    app.client = Mock()
    app.timer = Mock()
    app.restore_timer = Mock()
    app.home = Mock()
    app.home.resume.side_effect = lambda: app.state.views(tmp_path)
    app.stack = [app.home]
    app.version = -1
    app.runs = {}
    app.lock = asyncio.Lock()
    yield app
    app.close()


@pytest.mark.parametrize("callback", ["restore_location", "refresh_data"])
@pytest.mark.parametrize("outcome", ["data", "unchanged", "error"])
def test_close_during_refresh_stops_ui_updates(interface, monkeypatch, callback, outcome):
    async def scenario():
        entered = asyncio.Event()
        released = asyncio.Event()

        async def read_data(*args):
            entered.set()
            await released.wait()
            if outcome == "error":
                raise OSError("read failed after disconnect")
            return (1, {}, []) if outcome == "data" else None

        notify = Mock()
        monkeypatch.setattr(driver.run, "io_bound", read_data)
        monkeypatch.setattr(driver.ui, "notify", notify)
        task = asyncio.create_task(getattr(interface, callback)())
        await entered.wait()
        interface.close()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        interface.home.resume.assert_not_called()
        interface.home.refresh_data.assert_not_called()
        interface.client.run_javascript.assert_not_called()
        notify.assert_not_called()
        assert interface.version == -1

    asyncio.run(scenario())


def test_restore_waiting_for_refresh_lock_stops_after_close(interface, monkeypatch):
    async def scenario():
        read_data = AsyncMock()
        monkeypatch.setattr(driver.run, "io_bound", read_data)
        async with interface.lock:
            task = asyncio.create_task(interface.restore_location())
            await asyncio.sleep(0)
            interface.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        read_data.assert_not_called()
        interface.home.resume.assert_not_called()
        interface.client.run_javascript.assert_not_called()

    asyncio.run(scenario())


def test_close_cancels_both_timers_once(interface):
    interface.close()
    interface.close()
    interface.timer.cancel.assert_called_once_with(with_current_invocation=True)
    interface.restore_timer.cancel.assert_called_once_with(with_current_invocation=True)


def test_connected_refresh_still_updates_screens(interface, monkeypatch):
    monkeypatch.setattr(driver.run, "io_bound", AsyncMock(return_value=(1, {}, [])))
    asyncio.run(interface.refresh_data())
    assert interface.version == 1
    interface.home.refresh_data.assert_called_once_with()
