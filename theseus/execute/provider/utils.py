"""Shared primitives for remote execution providers."""

import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


#### results ####


@dataclass(frozen=True)
class RunResult:
    """Result of a remote command."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Return whether the remote command succeeded."""
        return self.returncode == 0


@dataclass(frozen=True)
class ShipResult:
    """Remote artifacts and process identities created for a dispatch."""

    provider: str
    host: str
    directory: str
    bootstrap: str
    dispatch: str
    logs: tuple[str, ...]
    job_ids: tuple[str, ...]
    remote: RunResult

    @property
    def ok(self) -> bool:
        """Return whether every requested remote launch was accepted."""
        return self.remote.ok and bool(self.job_ids)


#### remote execution ####


_host_last_call: dict[str, float] = {}
_host_interval: dict[str, float] = {}
_host_lock = threading.Lock()

_MIN_INTERVAL = 0.5
_MAX_INTERVAL = 10.0
_MAX_ATTEMPTS = 5


def uv_groups_for_cpu(groups: Iterable[str]) -> list[str]:
    """Replace an accelerator dependency flavor with the CPU flavor."""
    hardware_groups = {"cpu", "cuda12", "cuda13", "tpu"}
    return [*(group for group in groups if group not in hardware_groups), "cpu"]


def place(
    minimum: int,
    capacities: list[int],
    limit: int | None = None,
) -> tuple[int, ...] | None:
    """Return the smallest homogeneous machine layout satisfying a request."""
    capacities = sorted(
        (capacity for capacity in capacities if capacity > 0),
        reverse=True,
    )
    for machine_count in range(1, len(capacities) + 1):
        per_machine = (minimum + machine_count - 1) // machine_count
        total = per_machine * machine_count
        if limit is not None and total > limit:
            continue
        if sum(capacity >= per_machine for capacity in capacities) >= machine_count:
            return (per_machine,) * machine_count
    return None


def run(
    command: str | list[str],
    host: str,
    timeout: float | None = None,
    max_attempts: int | None = None,
) -> RunResult:
    """Run a command remotely through an SSH login shell.

    Transient connection failures are retried with per-host adaptive backoff.
    Pass ``max_attempts=1`` for commands whose remote effects must not be
    repeated.
    """
    if isinstance(command, list):
        command = " ".join(command)

    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        f"$SHELL -l -c {shlex.quote(command)}",
    ]
    preview = command[:80] + "..." if len(command) > 80 else command
    logger.debug("SSH | running on {}: {}", host, preview)
    return _run_subprocess(
        ssh_command,
        host,
        timeout,
        "SSH",
        _MAX_ATTEMPTS if max_attempts is None else max(1, max_attempts),
    )


def copy(
    source: str | Path,
    host: str,
    destination: str,
    timeout: float | None = None,
) -> RunResult:
    """Copy one local file or directory onto a remote host."""
    source = Path(source)
    if not source.exists():
        return RunResult(1, "", f"Copy source does not exist: {source}")

    parent = destination.rsplit("/", 1)[0] or "."
    prepared = run(f"mkdir -p -- {shlex.quote(parent)}", host, timeout=timeout)
    if not prepared.ok:
        return prepared

    staging = f"{destination}.partial-{uuid.uuid4().hex}"
    command = ["scp", "-o", "BatchMode=yes"]
    if source.is_dir():
        command.append("-r")
    command.extend((str(source), f"{host}:{staging}"))

    copied = _run_subprocess(command, host, timeout, "SCP", _MAX_ATTEMPTS)
    if not copied.ok:
        run(f"rm -rf -- {shlex.quote(staging)}", host, timeout=timeout)
        return copied

    quoted_staging = shlex.quote(staging)
    quoted_destination = shlex.quote(destination)
    if source.is_dir():
        publish = (
            f"rm -rf -- {quoted_destination} && "
            f"mv -T -- {quoted_staging} {quoted_destination}"
        )
    else:
        publish = f"mv -f -- {quoted_staging} {quoted_destination}"
    published = run(publish, host, timeout=timeout, max_attempts=1)
    if published.returncode == -1:
        verified = run(
            f"test ! -e {quoted_staging} && test -e {quoted_destination}",
            host,
            timeout=timeout,
        )
        if verified.ok:
            return verified
    return published


def _run_subprocess(
    command: list[str],
    host: str,
    timeout: float | None,
    label: str,
    attempts: int,
) -> RunResult:
    """Run a retryable transport subprocess with per-host backoff."""
    transient_errors = (
        "connection reset",
        "connection closed",
        "kex_exchange",
        "connection refused",
        "connection timed out",
        "no route to host",
    )
    result = RunResult(-1, "", "max attempts exceeded")
    for attempt in range(attempts):
        with _host_lock:
            now = time.time()
            interval = _host_interval.get(host, _MIN_INTERVAL)
            wait = interval - (now - _host_last_call.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            _host_last_call[host] = time.time()

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = RunResult(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode()
            result = RunResult(-1, stdout, f"{label} timed out after {timeout}s")

        if result.ok:
            with _host_lock:
                interval = _host_interval.get(host, _MIN_INTERVAL)
                _host_interval[host] = max(interval * 0.8, _MIN_INTERVAL)
            return result
        transient = any(error in result.stderr.lower() for error in transient_errors)
        if not transient or attempt == attempts - 1:
            logger.debug(
                "{} | command failed on {} (rc={}): {}",
                label,
                host,
                result.returncode,
                result.stderr[:200] or "no stderr",
            )
            return result
        with _host_lock:
            interval = _host_interval.get(host, _MIN_INTERVAL)
            _host_interval[host] = min(interval * 2.0, _MAX_INTERVAL)
        logger.warning(
            "{} | transient error on {} (attempt {}/{}), backing off",
            label,
            host,
            attempt + 1,
            attempts,
        )
    return result
