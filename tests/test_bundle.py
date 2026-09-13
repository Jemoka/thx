import io
import subprocess
import tarfile
from pathlib import Path

from theseus.execute.bundle import bundle


def test_bundle_archives_clean_and_dirty_state_without_mutating_git(
    tmp_path: Path, monkeypatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@theseus.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Theseus Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    deleted = tmp_path / "deleted.txt"
    tracked.write_text("committed\n")
    deleted.write_text("remove me\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    tracked.write_text("dirty\n")
    deleted.unlink()
    (tmp_path / "source.py").write_text("untracked source\n")
    (tmp_path / "payload.bin").write_bytes(b"untracked binary")
    index = tmp_path / ".git" / "index"
    index_before = index.read_bytes()
    objects_before = subprocess.run(
        ["git", "count-objects", "-v"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    status_before = subprocess.run(
        ["git", "status", "--short"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    monkeypatch.chdir(tmp_path)

    clean = bundle(dirty=False)
    first = bundle(dirty=True)
    second = bundle(dirty=True)

    assert isinstance(clean, bytearray)
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(clean), mode="r:gz") as archive:
        assert archive.extractfile("tracked.py").read() == b"committed\n"
        assert "deleted.txt" in archive.getnames()
        assert "source.py" not in archive.getnames()
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert archive.extractfile("tracked.py").read() == b"dirty\n"
        assert "deleted.txt" not in archive.getnames()
        assert archive.extractfile("source.py").read() == b"untracked source\n"
        assert "payload.bin" not in archive.getnames()
    assert index.read_bytes() == index_before
    assert (
        subprocess.run(
            ["git", "count-objects", "-v"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == objects_before
    )
    assert (
        subprocess.run(
            ["git", "status", "--short"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == status_before
    )
