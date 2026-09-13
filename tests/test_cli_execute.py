"""Local CLI execution uses the same configured chain as submit."""

from dataclasses import dataclass
import json

from typer.testing import CliRunner

from theseus.cli.app import app
from theseus.config import configure, field
from theseus.job import BasicJob
from theseus.registry import job


@dataclass
class LocalConfig:
    value: int = field("fixture/value", default=3)


@job("tests/cli/local")
class LocalJob(BasicJob[LocalConfig]):
    @classmethod
    def config(cls):
        return LocalConfig

    def state_init(self):
        pass

    def state_restore(self, state):
        pass

    def run(self):
        with self.spec.result("value.json") as output:
            json.dump({"value": configure(LocalConfig).value}, output)


def test_cli_run_executes_an_attached_yaml_and_overrides(tmp_path):
    runner = CliRunner()
    path = tmp_path / "job.yaml"
    configured = runner.invoke(app, ["configure", "tests/cli/local", str(path)])
    assert configured.exit_code == 0, configured.output
    root = tmp_path / "results"
    result = runner.invoke(app, ["run", "local", str(path), str(root), "fixture.value=17"])
    assert result.exit_code == 0, result.output
    artifacts = list(root.rglob("value.json"))
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_text()) == {"value": 17}


def test_cli_run_dry_run_does_not_construct_a_job(tmp_path, monkeypatch):
    path = tmp_path / "job.yaml"
    runner = CliRunner()
    assert runner.invoke(app, ["configure", "tests/cli/local", str(path)]).exit_code == 0
    monkeypatch.setattr(LocalJob, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("constructed")))
    result = runner.invoke(app, ["run", "local", str(path), str(tmp_path / "unused"), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["hardware"] is None


def test_cli_run_rejects_invalid_overrides(tmp_path):
    path = tmp_path / "job.yaml"
    runner = CliRunner()
    assert runner.invoke(app, ["configure", "tests/cli/local", str(path)]).exit_code == 0
    result = runner.invoke(app, ["run", "local", str(path), str(tmp_path), "fixture.value=bad"])
    assert result.exit_code != 0


def test_cli_run_preserves_nonce_and_fresh_starts_again(tmp_path):
    from omegaconf import OmegaConf
    from theseus.store import ObjectReader

    runner = CliRunner()
    path = tmp_path / "job.yaml"
    assert runner.invoke(app, ["configure", "tests/cli/local", str(path)]).exit_code == 0
    original = path.read_bytes()
    nonce = OmegaConf.load(path).execution.nonce
    root = tmp_path / "results"
    command = ["run", "local", str(path), str(root)]
    for _ in range(2):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
        assert len(list(root.rglob("value.json"))) == 1
        assert len(ObjectReader.local(root).query().select()) == 1
    preview = runner.invoke(app, command + ["--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.stdout)["nonce"] == nonce
    assert json.loads(preview.stdout)["job"]["nonce"] == nonce
    result = runner.invoke(app, command + ["--fresh"])
    assert result.exit_code == 0, result.output
    rows = ObjectReader.local(root).query().select(raw=True)
    assert len(rows) == 2
    assert len({row["_x_execution"] for row in rows}) == 2
    assert nonce in {row["_x_execution"] for row in rows}
    assert path.read_bytes() == original
