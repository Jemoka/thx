"""Real browser regression tests for terminal-style navigation and plot gestures.

Run `uv run playwright install chromium` before running this module.
"""

import math
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.request import urlopen

import pytest
from playwright.sync_api import expect, sync_playwright

from theseus.base import ExecutionSpec, Node
from theseus.cli.interface.state import InterfaceState
from theseus.store import ObjectStore


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("browser-runs")
    logs = root / "logs"
    logs.mkdir()
    logs.joinpath("alpha-group-train-abcdef.0.log").write_text(
        "[bootstrap] starting\n\x1b[36m2026-09-05 12:00:00\x1b[0m | INFO | rank zero\n"
    )
    logs.joinpath("alpha-group-train-abcdef.1.log").write_text(
        "2026-09-05 12:00:01 | INFO | rank one\n"
    )
    logs.joinpath("alpha-group-train-fedcba.0.log").write_text("other run\n")
    writer = ObjectStore(ExecutionSpec.local(str(root), name="ui").hardware)
    for nonce, factor in [("abcdef", 1), ("fedcba", 0.8)]:
        for seq in range(100):
            writer.value(
                Node(name="alpha.group.train", nonce=nonce, seq=seq),
                {
                    "step": seq,
                    "train/loss": factor
                    * (5 * math.exp(-seq / 25) + 0.1 * math.sin(seq)),
                    "tokens": seq * 128,
                    **({"_x_checkpoint": True} if seq in (10, 40, 80) else {}),
                },
            )
    writer.close()
    state = InterfaceState(root / "state" / "theseus" / "ui.sqlite3")
    view = state.add_view(root, "alpha.group.train", "abcdef")
    plot = state.add_plot(view)
    state.update_plot(
        plot,
        x="step",
        y="train/loss",
        log_x=False,
        log_y=False,
        checkpoints=True,
        smoothing=0.25,
    )
    state.close()
    return root


@pytest.fixture(scope="module")
def server(workspace):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = workspace / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "theseus.cli.app",
                "ui",
                "--serve",
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                str(workspace),
                "--logs",
                str(workspace / "logs"),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **{
                    key: value
                    for key, value in os.environ.items()
                    if key != "PYTEST_CURRENT_TEST"
                },
                "XDG_STATE_HOME": str(workspace / "state"),
                "XDG_CACHE_HOME": str(workspace / "cache"),
            },
        )
        try:
            deadline = monotonic() + 20
            while monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                try:
                    with urlopen(url, timeout=0.5):
                        break
                except OSError:
                    sleep(0.1)
            else:
                pytest.fail(log_path.read_text())
            yield url
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert "Traceback" not in log_path.read_text(), log_path.read_text()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(
        viewport={"width": 1200, "height": 800},
        permissions=["clipboard-read", "clipboard-write"],
    )
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    yield page
    context.close()
    assert not errors, errors


def test_home_keyboard_from_body_and_search_and_marks(page, server):
    page.goto(server)
    expect(page.get_by_role("treeitem").filter(has_text="alpha")).to_be_visible()
    page.locator("body").click(position={"x": 1000, "y": 500})
    page.keyboard.press("ArrowDown")
    expect(page.get_by_role("treeitem").filter(has_text="alpha")).to_be_focused()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowDown")
    expect(page.get_by_role("treeitem").filter(has_text="group")).to_be_focused()
    page.keyboard.press("m")
    expect(page.locator(".marked-runs")).to_contain_text("abcdef")
    expect(page.locator(".marked-runs")).to_contain_text("fedcba")
    page.keyboard.press("u")
    expect(page.locator(".workspace")).to_be_hidden()
    page.get_by_role("textbox", name="search", exact=True).click()
    page.keyboard.press("ArrowDown")
    expect(page.get_by_role("treeitem").filter(has_text="group")).to_be_focused()
    page.keyboard.press("Home")
    page.keyboard.press("ArrowUp")
    expect(page.get_by_role("textbox", name="search", exact=True)).to_be_focused()


def test_run_keyboard_from_fields_views_and_blank_space_and_refresh(page, server):
    page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=30")
    expect(page.get_by_text("seq 30", exact=True)).to_be_visible()
    page.get_by_text("tokens", exact=True).click()
    page.keyboard.press("ArrowRight")
    expect(page.get_by_text("seq 31", exact=True)).to_be_visible()
    # Focus a view using ordinary Tab navigation, without opening it.
    page.keyboard.press("Tab")
    page.keyboard.press("Tab")
    page.keyboard.press("Shift+ArrowRight")
    expect(page.get_by_text("seq 40", exact=True)).to_be_visible()
    page.locator("body").click(position={"x": 1100, "y": 450})
    page.keyboard.press("Shift+ArrowLeft")
    expect(page.get_by_text("seq 10", exact=True)).to_be_visible()
    page.get_by_role("button", name="jump", exact=True).click()
    editor = page.get_by_role("textbox", name="seq", exact=True)
    editor.fill("35")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("q")
    expect(page.get_by_text("seq 10", exact=True)).to_be_visible()
    editor.fill("35")
    page.keyboard.press("Enter")
    expect(page.get_by_text("seq 35", exact=True)).to_be_visible()
    expect(page).to_have_url(re.compile("seq=35"))
    page.reload()
    expect(page.get_by_text("seq 35", exact=True)).to_be_visible()
    page.locator(".node-identity:visible").click()
    expect(page.locator(".node-identity:visible")).to_have_class(re.compile("copied"))
    assert (
        page.evaluate("navigator.clipboard.readText()")
        == Node(name="alpha.group.train", nonce="abcdef", seq=35).serialize()
    )


def test_run_log_list_and_streaming_view(page, server, workspace):
    page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=30")
    expect(page.get_by_role("button", name="logs", exact=True)).to_be_visible()
    header = page.locator(".screen:not(.hidden) .run-top .row-line").first
    expect(header.locator("button")).to_have_count(4)
    assert header.locator("button").all_inner_texts() == ["‹", "logs", "walk", "jump"]

    page.get_by_role("button", name="logs", exact=True).click()
    files = page.locator(".log-list button")
    expect(files).to_have_count(2)
    expect(files.nth(0)).to_have_attribute(
        "aria-label", "alpha-group-train-abcdef.0.log"
    )
    expect(files.nth(0)).to_be_focused()
    page.keyboard.press("ArrowDown")
    expect(files.nth(1)).to_be_focused()
    page.keyboard.press("Enter")

    expect(
        page.get_by_text("alpha-group-train-abcdef.1.log", exact=True)
    ).to_be_visible()
    expect(page.get_by_text("1", exact=True)).to_be_visible()
    expect(
        page.get_by_text("2026-09-05 12:00:01 | INFO | rank one", exact=True)
    ).to_be_visible()
    expect(page).to_have_url(re.compile("screen=log"))

    with workspace.joinpath("logs/alpha-group-train-abcdef.1.log").open("a") as log:
        log.write("2026-09-05 12:00:02 | INFO | streamed line\n")
    expect(page.get_by_text("2", exact=True)).to_be_visible(timeout=5000)
    expect(
        page.get_by_text("2026-09-05 12:00:02 | INFO | streamed line", exact=True)
    ).to_be_visible()

    with workspace.joinpath("logs/alpha-group-train-abcdef.1.log").open("a") as log:
        log.writelines(
            f"2026-09-05 12:01:{line:02d} | INFO | line {line}\n" for line in range(60)
        )
    expect(
        page.get_by_text("2026-09-05 12:01:59 | INFO | line 59", exact=True)
    ).to_be_visible(timeout=5000)
    scroll = page.locator(".log-view .q-scrollarea__container")
    page.wait_for_function(
        "() => document.querySelector('.log-view .q-scrollarea__container').scrollTop > 0"
    )
    at_bottom = scroll.evaluate("e => e.scrollTop")
    page.keyboard.press("PageUp")
    page.wait_for_function(
        "before => document.querySelector('.log-view .q-scrollarea__container').scrollTop < before",
        arg=at_bottom,
    )
    page.keyboard.press("End")
    page.wait_for_function(
        "() => {const e=document.querySelector('.log-view .q-scrollarea__container');"
        "return e.scrollTop+e.clientHeight>=e.scrollHeight}"
    )
    page.reload()
    expect(page.get_by_text("following", exact=True)).to_be_visible()
    expect(
        page.get_by_text("2026-09-05 12:01:59 | INFO | line 59", exact=True)
    ).to_be_visible()
    page.keyboard.press("q")
    expect(files.nth(0)).to_be_visible()


def test_plot_gestures_controls_and_drilldown(page, server):
    page.goto(server + "/?screen=plot&run=alpha.group.train&nonce=abcdef&view=1&plot=1")
    overlay = page.locator(".chart-gestures:visible")
    expect(overlay).to_be_visible()
    chart = page.locator(".screen:not(.hidden) .js-plotly-plot")
    assert chart.evaluate("e=>e.data[0].type") == "scatter"
    assert chart.evaluate("e=>e.data[0].mode") == "lines"
    box = overlay.bounding_box()
    page.mouse.move(box["x"] + 0.2 * box["width"], box["y"] + 0.2 * box["height"])
    page.mouse.down()
    page.mouse.move(
        box["x"] + 0.8 * box["width"], box["y"] + 0.8 * box["height"], steps=8
    )
    page.mouse.up()
    page.wait_for_function(
        'document.querySelector(".screen:not(.hidden) .js-plotly-plot")._fullLayout.xaxis.range[0]>10'
    )
    overlay.click(position={"x": 20, "y": 20})
    page.wait_for_function(
        'document.querySelector(".screen:not(.hidden) .js-plotly-plot")._fullLayout.xaxis.range[0]===0'
    )
    page.mouse.move(box["x"] + 0.5 * box["width"], box["y"] + 0.5 * box["height"])
    page.mouse.down()
    page.wait_for_timeout(400)
    page.mouse.up()
    expect(page.locator(".plot-footer:visible button")).to_be_visible()
    expect(page.locator(".plot-footer:visible")).to_contain_text("TWEMA")
    page.locator(".plot-footer:visible button").click()
    expect(page.get_by_text("seq 49", exact=True)).to_be_visible()
    page.keyboard.press("q")
    expect(overlay).to_be_visible()
    page.get_by_role("button", name="x: step", exact=True).click()
    page.keyboard.press("Escape")
    expect(page.locator(".axis-options:visible")).to_have_count(0)
    expect(overlay).to_be_visible()
    page.get_by_role("button", name="○ log x", exact=True).click()
    page.wait_for_function(
        'document.querySelector(".screen:not(.hidden) .js-plotly-plot")._fullLayout.xaxis.type==="log"'
    )
    page.get_by_role("button", name="● log x", exact=True).click()
    page.get_by_role("slider", name="smoothing", exact=True).click()
    page.keyboard.press("Home")
    expect(page.get_by_text("smooth 0.00", exact=True)).to_be_visible()
    page.keyboard.press("ArrowRight")
    expect(page.get_by_text("smooth 0.05", exact=True)).to_be_visible()
    page.reload()
    expect(page.get_by_text("smooth 0.05", exact=True)).to_be_visible()


def test_multiway_comparison_and_url_restoration(page, server):
    page.goto(server)
    for label in ("▶ alpha", "▶ group", "▶ train"):
        page.get_by_role("treeitem").filter(has_text=label).click()
    for nonce in ("abcdef", "fedcba"):
        page.get_by_role("treeitem").filter(has_text=nonce).hover()
        page.keyboard.press("m")
        expect(page.locator(".marked-runs")).to_contain_text(nonce)
    page.reload()
    expect(page.locator(".marked-runs")).to_contain_text("fedcba")
    page.locator(".workspace").get_by_role("button", name="+", exact=True).click()
    expect(page.get_by_role("button", name="rename", exact=True)).to_be_visible()
    page.get_by_role("button", name="+", exact=True).click()
    for axis, value in [("x", "step"), ("y", "train/loss")]:
        page.get_by_role("button", name=f"{axis}: —", exact=True).click()
        page.locator(".axis-options:visible").get_by_role(
            "button", name=value, exact=True
        ).click()
    expect(page.locator(".chart-heading:visible")).to_contain_text("fedcba")
    url = page.url
    page.reload()
    expect(page.locator(".chart-heading:visible")).to_contain_text("abcdef")
    expect(page.locator(".chart-heading:visible")).to_contain_text("fedcba")
    expect(page).to_have_url(url)
    page.get_by_role("button", name="‹", exact=True).click()
    expect(page.get_by_role("button", name="walk", exact=True)).to_be_visible()
    page.go_back()
    expect(page.get_by_role("button", name="y: train/loss", exact=True)).to_be_visible()


def test_walk_renders_real_pygwalker_with_run_data(page, server):
    page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=30")
    page.get_by_role("button", name="walk", exact=True).click()
    walker = page.frame_locator("iframe[title=Pygwalker]").frame_locator("iframe")
    expect(walker.get_by_text("Field List", exact=True)).to_be_visible(timeout=30000)
    expect(walker.get_by_role("button", name="train/loss", exact=True)).to_be_visible()
    expect(walker.get_by_role("button", name="_x_nonce", exact=True)).to_be_visible()
    expect(walker.get_by_text("Visualization", exact=True)).to_be_visible()


def test_live_refresh_preserves_focus_fields_and_selection(page, server, workspace):
    page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=30")
    expect(page.get_by_text("seq 30", exact=True)).to_be_visible()
    page.get_by_text("tokens", exact=True).click()
    field = page.locator("details").filter(has=page.get_by_text("tokens", exact=True))
    expect(field).to_have_attribute("open", "")
    writer = ObjectStore(ExecutionSpec.local(str(workspace), name="ui").hardware)
    writer.value(
        Node(name="alpha.group.train", nonce="abcdef", seq=100),
        {"step": 100, "train/loss": 0.05},
    )
    writer.close()
    expect(page.get_by_role("slider", name="run timeline")).to_have_attribute(
        "aria-valuemax", "100", timeout=10000
    )
    expect(page.get_by_text("seq 30", exact=True)).to_be_visible()
    expect(field).to_have_attribute("open", "")
    page.keyboard.press("ArrowRight")
    expect(page.get_by_text("seq 31", exact=True)).to_be_visible()


def test_checkpoint_double_click_opens_exact_run_sequence(page, server):
    page.goto(server + "/?screen=plot&run=alpha.group.train&nonce=abcdef&view=1&plot=1")
    overlay = page.locator(".chart-gestures:visible")
    expect(overlay).to_be_visible()
    position = page.locator(".screen:not(.hidden) .js-plotly-plot").evaluate("""e => {
        const trace=e.data.find(t=>t.name==='checkpoint');
        const {xaxis:x,yaxis:y}=e._fullLayout;
        return {x:(trace.x[0]-x.range[0])/(x.range[1]-x.range[0])*x._length,
                y:(y.range[1]-trace.y[0])/(y.range[1]-y.range[0])*y._length};
    }""")
    overlay.dblclick(position=position)
    expect(page.get_by_text("seq 10", exact=True)).to_be_visible()
    expect(page.locator(".checkpoint-label:visible")).to_contain_text("checkpoint")
    expect(page).to_have_url(re.compile("seq=10"))


def test_comparison_inspection_wraps_without_overlapping(page, server):
    page.goto(server)
    for label in ("▶ alpha", "▶ group", "▶ train"):
        page.get_by_role("treeitem").filter(has_text=label).click()
    page.keyboard.press("m")
    expect(page.locator(".marked-runs")).to_contain_text("fedcba")
    page.locator(".workspace").get_by_role("button", name="view 1", exact=True).click()
    page.get_by_role("button", name="plot 1", exact=True).click()
    page.set_viewport_size({"width": 480, "height": 700})
    overlay = page.locator(".chart-gestures:visible")
    expect(overlay).to_be_visible()
    page.wait_for_timeout(200)
    box = overlay.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.wait_for_timeout(400)
    page.mouse.up()
    footer = page.locator(".plot-footer:visible")
    expect(footer.locator("button")).to_have_count(2)
    dimensions = footer.evaluate(
        """e=>({height:e.clientHeight,width:e.clientWidth,scroll:e.scrollWidth,boxes:[...e.children].map(c=>{const r=c.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})})"""
    )
    assert dimensions["height"] > 20
    assert (
        footer.bounding_box()["y"] + dimensions["height"]
        <= page.viewport_size["height"]
    )
    assert dimensions["scroll"] <= dimensions["width"]
    boxes = dimensions["boxes"]
    for a, b in zip(boxes, boxes[1:]):
        assert b["y"] >= a["y"] + a["h"] - 0.5 or b["x"] >= a["x"] + a["w"] - 0.5


def test_inline_rename_and_confirmed_delete_keep_navigation_intact(page, server):
    page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=30")
    page.get_by_role("button", name="+", exact=True).click()
    expect(page.get_by_role("button", name="rename", exact=True)).to_be_visible()
    page.get_by_role("button", name="rename", exact=True).click()
    page.get_by_role("textbox", name="name", exact=True).fill("typed view")
    page.keyboard.press("Enter")
    expect(page.get_by_text("typed view", exact=True)).to_be_visible()
    page.get_by_role("button", name="+", exact=True).click()
    expect(page.get_by_role("button", name="x: —", exact=True)).to_be_visible()
    page.get_by_role("button", name="delete", exact=True).click()
    expect(page.get_by_role("button", name="x: —", exact=True)).to_be_visible()
    page.get_by_role("button", name="confirm", exact=True).click()
    expect(page.get_by_text("typed view", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="plot 1", exact=True)).to_have_count(0)
    page.get_by_role("button", name="delete", exact=True).click()
    page.get_by_role("button", name="confirm", exact=True).click()
    expect(page.get_by_role("button", name="jump", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="typed view", exact=True)).to_have_count(0)


def test_run_status_refreshes_from_logs_without_new_store_rows(page, server, workspace):
    logs = [workspace / "logs" / f"alpha-group-train-abcdef.{rank}.log" for rank in (0, 1)]
    saved = [path.read_bytes() for path in logs]
    try:
        for path in logs:
            path.write_text("[bootstrap] dispatch nonce=abcdef\n")
        page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=30")
        badge = page.locator(".run-status:visible")
        expect(badge).to_have_text("running", timeout=15000)
        for path in logs:
            with path.open("a") as output:
                output.write("[bootstrap] cleanup started with exit code 0\n")
        expect(badge).to_have_text("completed", timeout=15000)
        with logs[1].open("a") as output:
            output.write("[bootstrap] cleanup started with exit code 1\n")
        expect(badge).to_have_text("failed", timeout=15000)
    finally:
        for path, content in zip(logs, saved):
            path.write_bytes(content)


def test_run_checkpoint_opens_saved_configuration_and_job(page, server, workspace):
    writer = ObjectStore(ExecutionSpec.local(str(workspace), name="ui").hardware)
    with writer.blob(Node(name="alpha.group.train", nonce="abcdef", seq=10), {"_x_checkpoint": True}) as directory:
        (directory / "config.yaml").write_text("architecture:\n  block_size: 128\n")
        (directory / "job.json").write_text('{"name": "checkpoint-fixture"}')
    writer.close()
    page.goto(server + "/?screen=run&run=alpha.group.train&nonce=abcdef&seq=10")
    page.get_by_role("button", name="checkpoint", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("block_size")
    expect(dialog).to_contain_text("checkpoint-fixture")
    dialog.get_by_role("button", name="close", exact=True).click()
    expect(dialog).to_be_hidden()
