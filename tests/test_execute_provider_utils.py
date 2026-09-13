from pathlib import Path
from types import SimpleNamespace

import theseus.execute.provider.utils as provider_utils
from theseus.execute.provider.utils import RunResult, copy


def test_copy_publishes_a_directory_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "dispatch"
    source.mkdir()
    (source / "dispatch.json").write_text("{}")
    processes: list[list[str]] = []
    commands: list[tuple[str, int | None]] = []

    def fake_process(command, host, timeout, label, attempts):
        processes.append(command)
        return RunResult(0, "", "")

    def fake_run(command, host, timeout=None, max_attempts=None):
        commands.append((command, max_attempts))
        return RunResult(0, "", "")

    monkeypatch.setattr(
        provider_utils.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="shipment"),
    )
    monkeypatch.setattr(provider_utils, "_run_subprocess", fake_process)
    monkeypatch.setattr(provider_utils, "run", fake_run)

    result = copy(
        source,
        "login",
        "/share/project/run/nonce",
        timeout=12,
    )

    assert result.ok
    assert processes == [
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-r",
            str(source),
            "login:/share/project/run/nonce.partial-shipment",
        ]
    ]
    assert commands[0] == ("mkdir -p -- /share/project/run", None)
    assert "rm -rf -- /share/project/run/nonce" in commands[1][0]
    assert "mv -T" in commands[1][0]
    assert commands[1][1] == 1


def test_copy_rejects_a_missing_source(tmp_path: Path) -> None:
    result = copy(tmp_path / "missing", "login", "/share/missing")

    assert not result.ok
    assert "does not exist" in result.stderr
