import subprocess
from pathlib import Path

import pytest

import theseus.execute.bootstrap as bootstrap


def test_generate_embeds_only_current_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap, "bundle", lambda: bytearray(b"repository"))

    script = bootstrap.generate()
    generated = tmp_path / "bootstrap.sh"
    generated.write_text(script)

    assert "cmVwb3NpdG9yeQ==" in script
    assert "__REPO_BUNDLE__" not in script
    assert "__DISPATCH_SPEC__" not in script
    assert "usage: bootstrap.sh dispatchspec.json" in script
    subprocess.run(["bash", "-n", generated], check=True)
