"""External jobs survive CLI configuration and execution without their source."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CLI = ("uv", "run", "--no-sync", "--project", str(ROOT), "theseus")
JOB_SOURCE = """
import json
from dataclasses import dataclass
from pathlib import Path
from theseus.config import field
from theseus.job import BasicJob
from theseus.registry import job

@dataclass
class Config:
    value: int = field("external/value", default=3)
    events: str = field("external/events", default="events.jsonl")
    interrupt: bool = field("external/interrupt", default=True)

def measure_value(value):
    return value * 2

@job("external/train")
class ExternalJob(BasicJob[Config]):
    @classmethod
    def config(cls):
        return Config

    def state_init(self):
        self.total = measure_value(self.args.value)

    def state_restore(self, state):
        self.total = int((state["blob"] / "total").read_text())

    def run(self):
        self.total += measure_value(self.args.value)
        with self.store.blob(self.node, {"_x_checkpoint": True, "_x_job": self.JOB_NAME}) as blob:
            (blob / "total").write_text(str(self.total))
        with Path(self.args.events).open("a") as events:
            events.write(json.dumps({"total": self.total, "restored": self.base is not None,
                                     "tag": self.spec.tag}) + "\\n")
        if self.base is None and self.args.interrupt:
            raise RuntimeError("deliberate interruption after checkpoint")
"""


def run_process(*arguments: str, cwd: Path, succeeds: bool = True):
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT), "JAX_PLATFORMS": "cpu"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("module_form", ["dotted", "file"])
def test_external_job_runs_and_resumes_without_source(tmp_path, module_form):
    source = tmp_path / "custom" / "jobs.py"
    source.parent.mkdir()
    (source.parent / "__init__.py").write_text("")
    source.write_text(JOB_SOURCE)
    imported = "custom.jobs" if module_form == "dotted" else str(source)
    configured = tmp_path / "configured.yaml"
    run_process(
        *CLI,
        "--import",
        imported,
        "configure",
        "external/train",
        str(configured),
        "external.value=5",
        cwd=tmp_path,
    )
    assert OmegaConf.load(configured).execution.steps[0].job.type == "cloudpickle"
    # A fresh process must load both the job and its dataclass/helper from the YAML.
    source.unlink()
    edited = tmp_path / "edited.yaml"
    run_process(
        *CLI,
        "configure",
        str(configured),
        str(edited),
        "external.value=7",
        cwd=tmp_path,
    )
    assert OmegaConf.load(edited).execution.steps[0].job.type == "cloudpickle"
    run_process(
        sys.executable,
        "-c",
        """
from pathlib import Path
from omegaconf import OmegaConf
from theseus.execute.combobulator import Combobulation
execution = Combobulation.deserialize(OmegaConf.load("edited.yaml"))
execution = execution.resume(execution.jobs[0])
OmegaConf.save(execution.serialize(), "chain.yaml")
""",
        cwd=tmp_path,
    )
    dispatch = run_process(
        *CLI,
        "run",
        "external",
        "chain.yaml",
        "results",
        "--dry-run",
        cwd=tmp_path,
    )
    (tmp_path / "dispatch.json").write_text(dispatch.stdout)
    execute = """
from pathlib import Path
from theseus.execute.dispatch import DispatchSpec
dispatch = DispatchSpec.model_validate_json(Path("dispatch.json").read_text())
assert dispatch.run("results")
"""
    failed = run_process(sys.executable, "-c", execute, cwd=tmp_path, succeeds=False)
    assert "deliberate interruption after checkpoint" in failed.stderr
    run_process(sys.executable, "-c", execute, cwd=tmp_path)
    events = (tmp_path / "events.jsonl").read_text()
    assert [json.loads(line) for line in events.splitlines()] == [
        {"total": 28, "restored": False, "tag": "0:external/train"},
        {"total": 42, "restored": True, "tag": "0:external/train"},
        {"total": 56, "restored": True, "tag": "1:external/train"},
    ]
    run_process(sys.executable, "-c", execute, cwd=tmp_path)
    assert (tmp_path / "events.jsonl").read_text() == events
    run_process(
        *CLI,
        "run",
        "external-cli",
        "chain.yaml",
        "cli-results",
        "external.interrupt=false",
        "external.events=cli-events.jsonl",
        cwd=tmp_path,
    )
    assert [
        json.loads(line)
        for line in (tmp_path / "cli-events.jsonl").read_text().splitlines()
    ] == [
        {"total": 28, "restored": False, "tag": "0:external/train"},
        {"total": 42, "restored": True, "tag": "1:external/train"},
    ]


def test_imports_repeat_and_preserve_distinct_file_paths(tmp_path):
    first = tmp_path / "one" / "jobs.py"
    second = tmp_path / "two" / "jobs.py"
    for index, path in enumerate((first, second)):
        path.parent.mkdir()
        path.write_text(
            JOB_SOURCE.replace("external/train", f"external/job{index}")
            + f'\nwith open("imports.txt", "a") as output: output.write("{index}")\n'
        )
    result = run_process(
        *CLI,
        "--import",
        str(first),
        "--import",
        str(second),
        "--import",
        str(first),
        "jobs",
        "external/",
        cwd=tmp_path,
    )
    assert "external/job0" in result.stdout
    assert "external/job1" in result.stdout
    assert (tmp_path / "imports.txt").read_text() == "01"


@pytest.mark.parametrize("source", ["missing_module", "./missing.py"])
def test_missing_import_reports_the_option(tmp_path, source):
    result = run_process(*CLI, "--import", source, "jobs", cwd=tmp_path, succeeds=False)
    assert "--import" in result.stderr
    assert "Traceback" not in result.stderr
