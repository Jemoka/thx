import io
import json
import os
import subprocess
import tarfile
import time
import shutil
import sys
from pathlib import Path
from typing import Callable

import pytest

import theseus.execute.bootstrap as bootstrap


@pytest.fixture
def render_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[int], tuple[Path, Path, Path, Path]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "uv-calls"
    commands = {
        "juicefs": 'printf \'juicefs %s\\n\' "$*" >> "$UV_CALLS"',
        "setsid": 'exec "$@"',
        "timeout": (
            'if [[ "${SIMULATE_UV_TIMEOUT_ONCE:-}" == 1 && '
            '"$*" == *"uv sync"* && ! -e "$UV_CALLS.timed-out" ]]; then\n'
            '  touch "$UV_CALLS.timed-out"\n'
            "  exit 124\n"
            "fi\n"
            'if [[ -n "${SIMULATE_COMPLETED_PROBE_AT:-}" && '
            '"$*" == *"per_device_batch_size=${SIMULATE_COMPLETED_PROBE_AT}"* ]]; then\n'
            "  trap 'exit 143' TERM\n"
            "  echo 'TRAIN | 1/10 | loss 1.0'\n"
            "  while true; do sleep 1; done\n"
            "fi\n"
            'if [[ -n "${SIMULATE_TIMEOUT_KILL_AT:-}" && '
            '"$*" == *"per_device_batch_size=${SIMULATE_TIMEOUT_KILL_AT}"* ]]; then\n'
            "  echo \"timeout: sending signal TERM to command 'uv'\" >&2\n"
            "  exit 137\n"
            "fi\n"
            "while [[ $1 == --* ]]; do shift; done\n"
            "shift\n"
            'exec "$@"'
        ),
        "uv": (
            "printf '%s|UV_CACHE_DIR=%s|CUSTOM_VALUE=%s|OPT=%s|PGLE=%s|RUNS=%s\\n' "
            '"$*" "${UV_CACHE_DIR:-}" "${CUSTOM_VALUE:-}" '
            '"${JAX_OPTIMIZATION_LEVEL:-}" "${JAX_ENABLE_PGLE:-}" '
            '"${JAX_PGLE_PROFILING_RUNS:-}" >> "$UV_CALLS"\n'
            '[[ "${SIMULATE_UV_FAILURE:-}" != 1 ]] || exit 1\n'
            'if [[ "${SIMULATE_UV_SYNC_FAILURE_ONCE:-}" == 1 && '
            '$1 == sync && ! -e "$UV_CALLS.sync-failed" ]]; then\n'
            '  touch "$UV_CALLS.sync-failed"\n'
            "  exit 1\n"
            "fi\n"
            '[[ $1 != run || "${SIMULATE_RUN_FAILURE:-}" != 1 ]] || exit 1\n'
            "if [[ $1 == python && $2 == find ]]; then\n"
            '  if [[ -n "${BROKEN_PYTHON_PATH:-}" && ! -e "$UV_CALLS.repaired" ]]; then\n'
            '    echo "$BROKEN_PYTHON_PATH"\n'
            '    exit 0\n'
            '  fi\n'
            '  if [[ "${SIMULATE_BROKEN_PYTHON:-}" == 1 && '
            '! -e "$UV_CALLS.repaired" ]]; then\n'
            "    command -v false\n"
            "  else\n"
            "    command -v python3\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            "if [[ $1 == python && $2 == install ]]; then\n"
            '  touch "$UV_CALLS.repaired"\n'
            "  exit 0\n"
            "fi\n"
            'if [[ $1 == cache && $2 == clean ]]; then\n'
            '  [[ "${SIMULATE_CACHE_BUSY:-}" != 1 ]] || exit 1\n'
            '  rm -f "$UV_CACHE_DIR/broken"\n'
            '  exit 0\n'
            'fi\n'
            'if [[ $1 == sync ]]; then\n'
            '  mkdir -p .venv/bin\n'
            '  ln -sf "$(command -v python3)" .venv/bin/python\n'
            '  if [[ "${SIMULATE_MALFORMED_ENV_ONCE:-}" == 1 && '
            '! -e "$UV_CALLS.malformed" ]]; then\n'
            '    touch "$UV_CALLS.malformed"\n'
            '    ln -sf "$(command -v false)" .venv/bin/python\n'
            '  fi\n'
            '  exit 0\n'
            'fi\n'
            'if [[ "$*" == *"per_device_batch_size=1024"* ]]; then\n'
            '  [[ -z "${SIMULATE_EFFECTIVE_BATCH:-}" ]] || '
            'echo "BATCHING | ${SIMULATE_EFFECTIVE_BATCH} batchsize/node"\n'
            '  echo "${SIMULATE_OOM_MESSAGE:-RESOURCE_EXHAUSTED}"\n'
            "  exit 1\n"
            "fi\n"
            "exit 0"
        ),
    }
    for name, body in commands.items():
        command = bin_dir / name
        command.write_text(f"#!/usr/bin/env bash\n{body}\n")
        command.chmod(0o755)

    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        contents = b"[project]\nname = 'packed'\nversion = '0.0.0'\n"
        info = tarfile.TarInfo("pyproject.toml")
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))

    monkeypatch.setattr(
        bootstrap, "bundle", lambda: bytearray(archive_bytes.getvalue())
    )
    template = bootstrap.generate()

    def render(batch_size: int) -> tuple[Path, Path, Path, Path]:
        work = tmp_path / f"work-{batch_size}"
        logs = tmp_path / f"logs-{batch_size}"
        root = tmp_path / f"root-{batch_size}"
        dispatch = {
            "project": "project",
            "group": "group",
            "name": "name",
            "nonce": "abc123",
            "hardware": {
                "hosts": [
                    {
                        "cluster": {
                            "root": str(root),
                            "work": str(work),
                            "log": str(logs),
                            "mount": "redis://juicefs.example/1",
                            "cache_size": "1024",
                            "cache_dir": "/cache/juicefs",
                            "all_squash": "1000:1000",
                        },
                        "env": {
                            "UV_CACHE_DIR": "/cache/uv",
                            "CUSTOM_VALUE": "two words",
                        },
                        "uv_groups": ["cuda"],
                    }
                ]
            },
            "job": {
                "config": (
                    f"training:\n  per_device_batch_size: {batch_size}\n"
                    "logging:\n  remote: true\n"
                )
            },
        }
        script = tmp_path / f"bootstrap-{batch_size}.sh"
        dispatch_path = tmp_path / f"dispatch-{batch_size}.json"
        script.write_text(template)
        dispatch_path.write_text(json.dumps(dispatch))
        return script, dispatch_path, calls, logs

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("UV_CALLS", str(calls))
    return render


def test_environment_template_precedes_dependency_sync() -> None:
    template = (
        Path(__file__).parents[1] / "theseus" / "execute" / "bootstrap.sh"
    ).read_text()

    dependencies = template.index("phase=dependencies syncing environment")
    assert template.index("phase=environment") < dependencies
    assert "sync_environment" in template[dependencies:]
    assert "https://d.juicefs.com/install" in template
    assert "api.github.com/repos/juicedata/juicefs" not in template
    assert "urllib.request" not in template
    assert "BASH_COMMAND" not in template
    assert "umask 077" in template
    assert template.index("umask 022") < template.index("#### autobatch ####")
    assert template.index("phase=environment") < template.index('PYTHON_BIN="$(uv_with_timeout python find --no-project')
    assert "missing_packages+=(jq)" in template
    assert ': "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.95}"' in template
    assert "NCCL_DEBUG:=" not in template


def test_bootstrap_python_uses_uv_storage_without_a_staging_override() -> None:
    template = (
        Path(__file__).parents[1] / "theseus" / "execute" / "bootstrap.sh"
    ).read_text()

    assert 'UV_CACHE_DIR="$BOOTSTRAP_STAGING' not in template
    assert 'UV_PYTHON_INSTALL_DIR="$BOOTSTRAP_STAGING' not in template
    assert "uv_with_timeout python install --reinstall 3.11" in template


def test_bootstrap_runs_packed_dispatch_and_writes_backup_log(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
) -> None:
    script, dispatch, calls, logs = render_bootstrap(4)

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)
    resumed = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "░▀█▀░█░█░█▀▀░█▀▀░█▀▀░█░█░█▀▀" in result.stdout
    for phase in (
        "staging",
        "tools",
        "dispatch",
        "repository",
        "storage",
        "logging",
        "environment",
        "dependencies",
        "autobatch",
        "execution",
    ):
        assert f"[bootstrap] phase={phase}" in result.stdout
    assert "run --no-sync python -m theseus.execute.run" in calls.read_text()
    assert "training.per_device_batch_size=" not in calls.read_text()
    python_find = next(
        call
        for call in calls.read_text().splitlines()
        if call.startswith("python find")
    )
    assert "UV_CACHE_DIR=/cache/uv|CUSTOM_VALUE=two words" in python_find
    sync_call = next(
        call for call in calls.read_text().splitlines() if call.startswith("sync ")
    )
    assert "--python " in sync_call
    assert "--no-default-groups" in sync_call
    assert "--group cuda|UV_CACHE_DIR=/cache/uv|CUSTOM_VALUE=two words" in sync_call
    assert (
        "juicefs --quiet mount --umask=0000 -d --cache-size 1024 "
        "--cache-dir /cache/juicefs/theseus-abc123-0"
    ) in calls.read_text()
    assert [log.name for log in logs.iterdir()] == ["project-group-name-abc123.0.log"]
    assert (
        logs.joinpath("project-group-name-abc123.0.log")
        .read_text()
        .count("[bootstrap] dispatch nonce=abc123")
        == 2
    )


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("overridden", [False, True])
def test_bootstrap_optimization_defaults(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    disabled: bool,
    overridden: bool,
) -> None:
    settings = {
        "JAX_OPTIMIZATION_LEVEL": "O3",
        "JAX_ENABLE_PGLE": "false",
        "JAX_PGLE_PROFILING_RUNS": "5",
    }
    for key in (*settings, "THESEUS_DISABLE_OPTIMIZATIONS"):
        monkeypatch.delenv(key, raising=False)
    script, dispatch, calls, _ = render_bootstrap(4)
    payload = json.loads(dispatch.read_text())
    env = payload["hardware"]["hosts"][0]["env"]
    if disabled:
        env["THESEUS_DISABLE_OPTIMIZATIONS"] = "1"
    if overridden:
        env.update(settings)
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    expected = "OPT=O3|PGLE=false|RUNS=5" if overridden else (
        "OPT=|PGLE=|RUNS=" if disabled else "OPT=O2|PGLE=true|RUNS=3"
    )
    run = next(line for line in calls.read_text().splitlines() if line.startswith("run "))
    assert run.endswith(expected)


def test_bootstrap_preserves_memory_only_juicefs_cache(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
) -> None:
    script, dispatch, calls, _ = render_bootstrap(4)
    payload = json.loads(dispatch.read_text())
    payload["hardware"]["hosts"][0]["cluster"]["cache_dir"] = "memory"
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--cache-dir memory" in calls.read_text()
    assert "memory/theseus-" not in calls.read_text()


@pytest.mark.parametrize("simulate_failure", [False, True])
def test_bootstrap_removes_its_juicefs_cache(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    tmp_path: Path,
    simulate_failure: bool,
) -> None:
    script, dispatch, _, _ = render_bootstrap(4)
    payload = json.loads(dispatch.read_text())
    cache_root = tmp_path / "juicefs-cache"
    cache = cache_root / "theseus-abc123-0"
    cache.mkdir(parents=True)
    (cache / "cached-block").write_text("cache")
    payload["hardware"]["hosts"][0]["cluster"]["cache_dir"] = str(cache_root)
    if simulate_failure:
        payload["hardware"]["hosts"][0]["env"]["SIMULATE_RUN_FAILURE"] = "1"
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    expected_returncode = 1 if simulate_failure else 0
    assert result.returncode == expected_returncode
    assert cache_root.exists()
    assert not cache.exists()
    assert "removing JuiceFS client cache" in result.stdout


def test_bootstrap_falls_back_from_an_unusable_tmpdir(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script, dispatch, _, _ = render_bootstrap(4)
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    monkeypatch.setenv("TMPDIR", str(blocked))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[bootstrap] phase=staging selecting temporary storage" in result.stdout
    assert f"[bootstrap] staging directory: {blocked}" not in result.stdout
    assert 'export TMPDIR="$BOOTSTRAP_STAGING"' in script.read_text()


def test_bootstrap_falls_back_from_an_unusable_work_root(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    tmp_path: Path,
) -> None:
    script, dispatch, _, _ = render_bootstrap(4)
    blocked = tmp_path / "not-a-work-directory"
    blocked.write_text("blocked")
    payload = json.loads(dispatch.read_text())
    payload["hardware"]["hosts"][0]["cluster"]["work"] = str(blocked)
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"configured work root is unavailable: {blocked}" in result.stdout
    assert "[bootstrap] using temporary work root:" in result.stdout


def test_bootstrap_repairs_an_incomplete_python_installation(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
) -> None:
    script, dispatch, calls, _ = render_bootstrap(4)
    payload = json.loads(dispatch.read_text())
    payload["hardware"]["hosts"][0]["env"]["SIMULATE_BROKEN_PYTHON"] = "1"
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[bootstrap] repairing Python 3.11 through uv" in result.stdout
    install = next(
        call
        for call in calls.read_text().splitlines()
        if call.startswith("python install")
    )
    assert "python install --reinstall 3.11" in install
    assert "UV_CACHE_DIR=/cache/uv|CUSTOM_VALUE=two words" in install


def test_bootstrap_does_not_mask_a_broken_uv_cache(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
) -> None:
    script, dispatch, calls, _ = render_bootstrap(4)
    payload = json.loads(dispatch.read_text())
    payload["hardware"]["hosts"][0]["env"]["SIMULATE_UV_FAILURE"] = "1"
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode != 0
    assert "[bootstrap] repairing Python 3.11 through uv" in result.stdout
    assert "UV_CACHE_DIR=/cache/uv|CUSTOM_VALUE=two words" in calls.read_text()
    assert "/theseus-bootstrap." not in calls.read_text()


@pytest.mark.parametrize("damage", ["deleted-interpreter", "missing-stdlib"])
def test_bootstrap_repairs_partially_deleted_python_storage(render_bootstrap, tmp_path, damage):
    import shlex

    script, dispatch, calls, _ = render_bootstrap(4)
    depot = tmp_path / "depot"
    depot.mkdir()
    (depot / "partial-download").write_bytes(b"incomplete")
    broken = depot / "python"
    if damage == "missing-stdlib":
        broken.write_text(
            '#!/bin/sh\nPYTHONHOME=' + shlex.quote(str(depot)) +
            ' exec ' + shlex.quote(sys.executable) + ' "$@"\n'
        )
        broken.chmod(0o755)
        failed = subprocess.run([broken, "-c", "import builtins"], capture_output=True)
        assert failed.returncode != 0
        assert b"encodings" in failed.stderr
    payload = json.loads(dispatch.read_text())
    payload["hardware"]["hosts"][0]["env"].update(
        BROKEN_PYTHON_PATH=str(broken), UV_CACHE_DIR=str(depot),
    )
    dispatch.write_text(json.dumps(payload))
    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "python install --reinstall 3.11" in calls.read_text()
    assert "run --no-sync python -m theseus.execute.run" in calls.read_text()
    assert "cache clean" not in calls.read_text()


@pytest.mark.parametrize(
    ("failure", "sync_calls"),
    [("SIMULATE_MALFORMED_ENV_ONCE", 2), ("SIMULATE_UV_TIMEOUT_ONCE", 1)],
)
def test_bootstrap_resets_uv_cache_and_retries_sync(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    tmp_path: Path,
    failure: str,
    sync_calls: int,
) -> None:
    script, dispatch, calls, _ = render_bootstrap(4)
    cache = tmp_path / "uv-cache"
    cache.mkdir()
    (cache / "broken").touch()
    payload = json.loads(dispatch.read_text())
    environment = payload["hardware"]["hosts"][0]["env"]
    environment["UV_CACHE_DIR"] = str(cache)
    environment[failure] = "1"
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cleaning UV cache with UV's in-use protection" in result.stdout + result.stderr
    assert not (cache / "broken").exists()
    assert (
        sum(line.startswith("sync ") for line in calls.read_text().splitlines())
        == sync_calls
    )


@pytest.mark.parametrize("busy", [False, True])
def test_bootstrap_preserves_cache_on_failure_or_active_lock(render_bootstrap, tmp_path, busy):
    script, dispatch, calls, _ = render_bootstrap(4)
    cache = tmp_path / "uv-cache"
    cache.mkdir()
    marker = cache / "broken"
    marker.touch()
    payload = json.loads(dispatch.read_text())
    env = payload["hardware"]["hosts"][0]["env"]
    env["UV_CACHE_DIR"] = str(cache)
    if busy:
        env.update(SIMULATE_UV_TIMEOUT_ONCE="1", SIMULATE_CACHE_BUSY="1")
    else:
        env["SIMULATE_UV_SYNC_FAILURE_ONCE"] = "1"
    dispatch.write_text(json.dumps(payload))
    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)
    assert result.returncode != 0
    assert marker.exists()
    assert "--force" not in calls.read_text()
    if not busy:
        assert "cache clean" not in calls.read_text()


def test_uv_timeout_stops_a_hung_process_and_retries(tmp_path):
    # The controller ships uutils; exercise the GNU timeout used on workers.
    (tmp_path / "timeout").symlink_to(shutil.which("gnutimeout") or shutil.which("timeout"))
    template = (Path(__file__).parents[1] / "theseus/execute/bootstrap.sh").read_text()
    functions = template[template.index("uv_with_timeout() {"):template.index("sync_environment() {")]
    uv = tmp_path / "uv"
    marker = tmp_path / "hung"
    uv.write_text(
        '#!/usr/bin/env bash\n'
        '[[ "$1" != cache ]] || exit 0\n'
        'if [[ ! -e "$HANG_MARKER" ]]; then\n'
        '  touch "$HANG_MARKER"\n'
        '  exec sleep 60\n'
        'fi\n'
    )
    uv.chmod(0o755)
    started = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", functions + '\nTHESEUS_UV_OPERATION_TIMEOUT_SECONDS=1\nuv_with_timeout sync'],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "HANG_MARKER": str(marker)},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert 1 <= time.monotonic() - started < 10
    assert "cleaning UV cache" in result.stderr


def test_real_uv_preserves_an_active_cache_lock(tmp_path):
    import fcntl

    uv = shutil.which("uv")
    assert uv is not None
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = cache / "keep"
    marker.touch()
    with (cache / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        result = subprocess.run(
            ["timeout", "1", uv, "cache", "clean", "--cache-dir", str(cache)],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0
        assert marker.exists()


def test_real_uv_concurrent_sync_recovers_partial_environment_and_deleted_cache(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    template = (Path(__file__).parents[1] / "theseus/execute/bootstrap.sh").read_text()
    functions = template[template.index("uv_with_timeout() {"):template.index("trap 'echo")]
    cache = tmp_path / "cache"
    projects = [tmp_path / "first", tmp_path / "second"]
    for project in projects:
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname="bootstrap-test"\nversion="0.0.0"\n'
            '[tool.uv]\npackage=false\n'
        )
        # A killed install can leave a venv directory without its interpreter.
        (project / ".venv").mkdir()
        (project / ".venv/pyvenv.cfg").write_text("incomplete")
    for attempt in range(2):
        if attempt:
            shutil.rmtree(cache)
        processes = [subprocess.Popen(
            ["bash", "-c", functions + '\nTHESEUS_UV_OPERATION_TIMEOUT_SECONDS=30\nUV_GROUP_ARGUMENTS=()\nsync_environment'],
            cwd=project, env={**os.environ, "UV_CACHE_DIR": str(cache), "PYTHON_BIN": sys.executable, "UV_OFFLINE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) for project in projects]
        for project, process in zip(projects, processes):
            stdout, stderr = process.communicate(timeout=40)
            assert process.returncode == 0, stdout + stderr
            subprocess.run([project / ".venv/bin/python", "-c", "import ssl, sqlite3, venv"], check=True)


@pytest.mark.parametrize(
    ("name", "value"),
    (("NOT-AN-ENVIRONMENT-NAME", "value"), ("VALID_NAME", "contains\0nul")),
)
def test_bootstrap_rejects_an_unsafe_environment(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    name: str,
    value: str,
) -> None:
    script, dispatch, _, _ = render_bootstrap(4)
    payload = json.loads(dispatch.read_text())
    payload["hardware"]["hosts"][0]["env"][name] = value
    dispatch.write_text(json.dumps(payload))

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode != 0
    assert "[bootstrap] ERROR: invalid dispatch environment" in result.stdout


def test_bootstrap_autobatch_isolates_probes_and_overrides_real_run(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script, dispatch, calls, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    run_calls = [
        line for line in calls.read_text().splitlines() if line.startswith("run")
    ]
    assert len(run_calls) == 3
    assert "dispatch-autobatch-1024.json" in run_calls[0]
    assert "training.per_device_batch_size=1024" in run_calls[0]
    assert "logging.remote=false" in run_calls[0]
    assert "dispatch-autobatch-512.json" in run_calls[1]
    assert "training.per_device_batch_size=512" in run_calls[1]
    assert "logging.remote=false" in run_calls[1]
    assert "dispatch.json training.per_device_batch_size=512" in run_calls[2]
    assert "logging.remote=false" not in run_calls[2]

    log = tmp_path / "logs--1" / "project-group-name-abc123.0.log"
    contents = log.read_text()
    assert "RESOURCE_EXHAUSTED" in contents
    assert "probing training.per_device_batch_size=512" in contents


def test_bootstrap_autobatch_waits_for_delayed_log_writer(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import shlex

    script, dispatch, calls, logs = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")
    real_tee = shutil.which("tee")
    assert real_tee is not None
    delayed_tee = tmp_path / "bin" / "tee"
    delayed_tee.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == */autobatch-*.log ]]; then sleep 2; fi\n'
        f'exec {shlex.quote(real_tee)} "$@"\n'
    )
    delayed_tee.chmod(0o755)

    result = subprocess.run(
        ["bash", script, dispatch], text=True, capture_output=True, timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "No such file or directory" not in output
    run_calls = [line for line in calls.read_text().splitlines() if line.startswith("run")]
    assert len(run_calls) == 3
    assert "dispatch.json training.per_device_batch_size=512" in run_calls[-1]
    assert "RESOURCE_EXHAUSTED" in (logs / "project-group-name-abc123.0.log").read_text()


def test_bootstrap_autobatch_skips_clamped_duplicate_candidates(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script, dispatch, calls, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("SIMULATE_EFFECTIVE_BATCH", "128")

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    run_calls = [
        line for line in calls.read_text().splitlines() if line.startswith("run")
    ]
    assert len(run_calls) == 3
    assert "training.per_device_batch_size=1024" in run_calls[0]
    assert "training.per_device_batch_size=64" in run_calls[1]
    assert "dispatch.json training.per_device_batch_size=64" in run_calls[2]

    log = tmp_path / "logs--1" / "project-group-name-abc123.0.log"
    contents = log.read_text()
    for candidate in (512, 256, 128):
        assert f"skipping duplicate effective batch candidate {candidate}" in contents


def test_bootstrap_autobatch_accepts_candidate_override(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, dispatch, calls, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("THESEUS_AUTOBATCH_SIZE_CANDIDATES", "8 4 2 1")

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    run_calls = [
        line for line in calls.read_text().splitlines() if line.startswith("run")
    ]
    assert len(run_calls) == 2
    assert "training.per_device_batch_size=8" in run_calls[0]
    assert "training.per_device_batch_size=8" in run_calls[1]


def test_bootstrap_autobatch_rejects_invalid_candidate_override(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, dispatch, _, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_AUTOBATCH_SIZE_CANDIDATES", "8 nope 2")

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 1
    assert "[bootstrap] ERROR: invalid autobatch candidate: nope" in result.stdout


def test_bootstrap_autobatch_recognizes_rematerialization_oom(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, dispatch, calls, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv(
        "SIMULATE_OOM_MESSAGE",
        "Can't reduce memory use below 43.64GiB by rematerialization",
    )

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    run_calls = [
        line for line in calls.read_text().splitlines() if line.startswith("run")
    ]
    assert len(run_calls) == 3
    assert "training.per_device_batch_size=512" in run_calls[-1]


def test_bootstrap_autobatch_accepts_a_completed_training_step(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, dispatch, calls, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("THESEUS_MAIN_CHILD_TERM_GRACE_SECONDS", "0")
    monkeypatch.setenv("SIMULATE_COMPLETED_PROBE_AT", "512")

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
    run_calls = [
        line for line in calls.read_text().splitlines() if line.startswith("run")
    ]
    assert len(run_calls) == 2
    assert "training.per_device_batch_size=1024" in run_calls[0]
    assert "dispatch.json training.per_device_batch_size=512" in run_calls[1]


def test_bootstrap_autobatch_rejects_a_probe_killed_after_its_timeout(
    render_bootstrap: Callable[[int], tuple[Path, Path, Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, dispatch, _, _ = render_bootstrap(-1)
    monkeypatch.setenv("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("SIMULATE_TIMEOUT_KILL_AT", "512")

    result = subprocess.run(["bash", script, dispatch], text=True, capture_output=True)

    assert result.returncode == 137
    assert "[bootstrap] ERROR: autobatch probe failed with exit 137" in result.stdout
