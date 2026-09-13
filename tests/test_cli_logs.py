from pathlib import Path

from theseus.cli.interface.data import RunKey
from theseus.cli.interface.log_data import LogFile, LogReader


def test_log_files_discovers_only_ranked_new_execute_logs(tmp_path: Path) -> None:
    expected = [
        tmp_path / "alpha-group-train_job-abcdef.0.log",
        tmp_path / "alpha-group-train_job-abcdef.2.log",
    ]
    for path in expected:
        path.write_text(path.name)
    for name in (
        "alpha-group-train_job-fedcba.0.log",
        "alpha_group_train_job_abcdef.log",
        "alpha-group-train_job-abcdef.worker.log",
    ):
        (tmp_path / name).write_text(name)

    files = LogFile.discover(tmp_path, RunKey("alpha.group.train/job", "abcdef"))

    assert [file.path for file in files] == expected
    assert [file.rank for file in files] == [0, 2]


def test_log_reader_tails_numbers_and_follows_partial_lines(tmp_path: Path) -> None:
    path = tmp_path / "run.0.log"
    path.write_bytes(b"first\nsecond\n\x1b[36mthird\x1b[0m\npartial")
    reader = LogReader(path, limit=2)

    initial = reader.read()

    assert initial.reset
    assert initial.lines == ((2, "second"), (3, "third"))
    assert initial.partial == (4, "partial")
    assert initial.total_lines == 4

    with path.open("ab") as log:
        log.write(b" line\nfifth\n")
    appended = reader.read()

    assert not appended.reset
    assert appended.lines == ((4, "partial line"), (5, "fifth"))
    assert appended.partial is None
    assert appended.total_lines == 5

    with path.open("ab") as log:
        log.write(b"sixth\nseventh\neighth\n")
    burst = reader.read()

    assert burst.lines == ((7, "seventh"), (8, "eighth"))
    assert burst.total_lines == 8

    path.write_text("replacement\n")
    replaced = reader.read()

    assert replaced.reset
    assert replaced.lines == ((1, "replacement"),)
    assert replaced.total_lines == 1


def test_log_discovery_uses_dispatch_identity_instead_of_run_nonce(tmp_path):
    for name in ("alpha-group-other-dispatch-123abc.0.log", "alpha-group-other-456abc.0.log", "alpha-group-other-fedcba.0.log"):
        (tmp_path / name).write_text("log\n")
    files = LogFile.discover(tmp_path, RunKey("alpha.group.train", "abcdef"), ("dispatch-123abc", "456abc"))
    assert {file.path.name for file in files} == {"alpha-group-other-dispatch-123abc.0.log", "alpha-group-other-456abc.0.log"}


import os
from types import SimpleNamespace
import pytest
from theseus.cli.interface.status import LogStatus


@pytest.mark.parametrize("end,age,state", [
    ("", 0, "running"),
    ("", 600, "stale"),
    ("[bootstrap] cleanup started with exit code 0\n", 600, "completed"),
    ("[bootstrap] cleanup started with exit code 1\n", 0, "failed"),
    ("[bootstrap] cleanup started with exit code 0\n[bootstrap] dispatch nonce=123abc\n", 0, "running"),
])
def test_run_status_reads_latest_execution_logs(tmp_path, end, age, state):
    from time import time
    data = SimpleNamespace(name="alpha.group.train", nonce="abcdef", executions=("older", "123abc"))
    (tmp_path / "alpha-group-train-older.0.log").write_text("[bootstrap] cleanup started with exit code 1\n")
    path = tmp_path / "alpha-group-train-123abc.0.log"
    path.write_text("[bootstrap] dispatch nonce=123abc\n" + end)
    os.utime(path, (time() - age, time() - age))
    assert LogStatus.read(tmp_path, data).state == state


def test_run_status_includes_other_rank_failure_and_missing_logs(tmp_path):
    data = SimpleNamespace(name="alpha.group.train", nonce="abcdef", executions=())
    assert LogStatus.read(tmp_path, data).state == "no logs"
    (tmp_path / "alpha-group-train-abcdef.0.log").write_text("[bootstrap] cleanup started with exit code 0\n")
    (tmp_path / "alpha-group-train-abcdef.1.log").write_text("[bootstrap] cleanup started with exit code 9\n")
    assert LogStatus.read(tmp_path, data).state == "failed"
    assert LogStatus.read(tmp_path / "absent", data).state == "unavailable"
