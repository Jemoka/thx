# Inspecting runs

Open the NiceGUI interface over an existing Theseus root:

```sh
uv run python -m theseus.cli.app ui /path/to/root
```

The old `theseus/web` dashboard has been removed. Run browsing, checkpoint
inspection, fields, and live logs now live in this interface.

Pass the execution log directory to enable log browsing and run status:

```sh
uv run python -m theseus.cli.app ui /path/to/root --logs /path/to/logs
```

To serve without opening a browser:

```sh
uv run python -m theseus.cli.app ui --serve --bind 127.0.0.1 --port 8000 /path/to/root
```

Restart the server after changing its Python code. The interface uses the same
`$XDG_STATE_HOME/theseus/ui.sqlite3` saved views as the previous terminal UI
(default: `~/.local/state/theseus/ui.sqlite3`); no view conversion is needed.

The URL records the current run, sequence, log, view or plot, marked comparison
runs, search, and expanded tree branches. Refreshing or sharing the URL restores
that location within the same root and saved-view database. Browser Back and
Forward also navigate between locations.

- **Home:** search by project, group, or nonce with a regular expression. Arrow
  keys navigate and expand the tree; Down enters it from search or empty space.
  Enter opens a run. `m` marks and `u` unmarks the hovered or selected row and
  its descendant runs. Marked runs reveal cross-cutting views below the tree.
- **Run:** Left/Right move between recorded steps and Shift+Left/Right move
  between checkpoints, including when fields or view rows have focus. `h`/`l`,
  PageUp/PageDown, and Home/End also scrub the timeline. Editable inputs keep
  their normal typing and cursor keys. `jump` accepts a sequence number;
  clicking the node identity copies it. On checkpoint steps, `checkpoint`
  opens the saved configuration and job metadata.
- **Status:** the run page and log list poll the latest execution logs every
  five seconds, independently of store updates. Bootstrap exit codes determine
  completed/failed; a log updated within five minutes is treated as running.
  Older logs show stale, which does not prove the process stopped. This is
  execution-level status inferred from logs, not a scheduler query.
- **Logs:** when `--logs` is configured, `logs` lists matching per-machine
  bootstrap logs for the current run. Select a file with the mouse or keyboard
  to open its numbered tail; the viewer follows appended lines automatically
  while leaving the viewport in place when you scroll upward.
- **Views and plots:** `+` creates an item, `rename` opens an inline editor,
  and `delete` requires a second `confirm` click. Tab and arrow keys operate
  controls. Escape closes an axis picker before leaving the screen; `q` or
  Escape otherwise returns to the preceding screen.
- **Plots:** drag a rectangle to zoom; click empty plot space to reset. Hold
  for 350 ms to inspect and drag while held to scrub the inspection. Hover or
  single-click a checkpoint for its values. Double-click a checkpoint to open
  its owning run at that exact sequence. Clicking an inspected value opens its
  run at the nearest recorded sequence. Comparison values wrap when needed.
- **Walk:** opens Pygwalker's interactive data and visualization tools over all
  rows of the run or comparison. Use its back control to close it.

Plots use Plotly with the terminal palette and custom Theseus gestures. Log
scales, checkpoint visibility, axes, and TWEMA smoothing remain saved per plot.
Finalized object-store parts refresh automatically without resetting selection.

For interface regression tests:

```sh
uv run playwright install chromium
uv run pytest tests/test_cli.py tests/test_cli_run.py tests/test_cli_browser.py
```
