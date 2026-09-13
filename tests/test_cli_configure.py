"""Configuration round trips and interactive editing use the same execution schema."""

import importlib
import json
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.cli.app import app
from theseus.cli.configure import load_configuration, save_configuration
from theseus.cli.configurator import Configurator
from theseus.cli.configurator.fields import edit_fields
from theseus.cli.configurator.steps import edit_steps
from theseus.config import field
from theseus.execute.combobulator import Combobulation, Combobulator
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.provider import SSHConfig, SSHProvider
from theseus.execute.solve import SolveResult, solve
from theseus.job import BasicJob
from theseus.registry import job


@dataclass
class CLIConfig:
    value: int = field("cli/value", default=3)


@job("tests/cli/first")
class CLIJob(BasicJob[CLIConfig]):
    @classmethod
    def config(cls):
        return CLIConfig


@dataclass
class OtherConfig:
    value: int = field("other/value", default=7)


@job("tests/cli/second")
class OtherJob(BasicJob[OtherConfig]):
    @classmethod
    def config(cls):
        return OtherConfig


def test_configure_job_and_execution_round_trip(tmp_path):
    path = tmp_path / "job.yaml"
    result = CliRunner().invoke(
        app,
        [
            "configure",
            "tests/cli/first",
            str(path),
            "cli.value=9",
            "--chip",
            "a100",
            "-n",
            "2",
            "--cluster",
            "one,two",
            "--exclude-cluster",
            "three",
        ],
    )
    assert result.exit_code == 0, result.output
    document = OmegaConf.load(path)
    assert document.execution.steps[0].job.type == "cloudpickle"
    assert "dispatch" not in document
    assert document.execution.preferred_clusters == ["one", "two"]
    assert document.execution.forbidden_clusters == ["three"]
    assert document.execution.sharding.tp == 1
    assert document.execution.sharding.activation_checkpointing is False
    chain = Combobulation.deserialize(document)
    assert chain.chips == ("a100-sxm4-80gb",)
    assert chain.gpus == 2
    assert chain.config.cli.value == 9
    document.execution = chain.branch(OtherJob).serialize().execution
    OmegaConf.save(document, path)
    output = tmp_path / "chain.yaml"
    result = CliRunner().invoke(
        app, ["configure", str(path), str(output), "other.value=11"]
    )
    assert result.exit_code == 0, result.output
    restored = Combobulation.deserialize(OmegaConf.load(output))
    assert restored.jobs == (CLIJob, OtherJob)
    assert restored.steps[1].base == "previous"
    assert restored.config.cli.value == 9
    assert restored.config.other.value == 11


@pytest.mark.parametrize(
    "extra", [["cli.typo=oops"], ["--previous", "old.yaml"], ["--tp", "0"]]
)
def test_configure_rejects_bad_input_without_writing(tmp_path, extra):
    output = tmp_path / "invalid.yaml"
    result = CliRunner().invoke(
        app, ["configure", "tests/cli/first", str(output), *extra]
    )
    assert result.exit_code != 0
    assert not output.exists()


def test_configure_execution_top_half(tmp_path):
    source = tmp_path / "execution.yaml"
    OmegaConf.save({"execution": {"steps": [{"job": "tests/cli/first"}]}}, source)
    target = tmp_path / "full.yaml"
    result = CliRunner().invoke(app, ["configure", str(source), str(target)])
    assert result.exit_code == 0, result.output
    document = OmegaConf.load(target)
    assert document.config.cli.value == 3
    assert "load_modules" not in document
    listed = CliRunner().invoke(app, ["jobs"])
    assert listed.exit_code == 0, listed.output
    assert "tests/cli/first" in listed.output


def test_configure_builtin_job_keeps_readable_name(tmp_path):
    path = tmp_path / "builtin.yaml"
    save_configuration(load_configuration("gpt/train/pretrain"), path)
    assert OmegaConf.load(path).execution.steps[0].job == "gpt/train/pretrain"


def test_python_configurator_cloudpickles_unregistered_jobs(tmp_path):
    class CustomJob(CLIJob):
        pass

    document = Combobulator().run(CustomJob).serialize()
    editor = Configurator(document)
    path = tmp_path / "python.yaml"
    editor.ask = Mock(side_effect=["Save", str(path)])
    assert editor.run() == path
    saved = OmegaConf.load(path)
    assert saved.execution.steps[0].job.type == "cloudpickle"
    assert "load_modules" not in saved
    dispatch = DispatchSpec(
        name="custom",
        project="test",
        group="default",
        hardware=HardwareResult(chip=None, hosts=[], total_chips=0),
        job=Combobulation.deserialize(saved).model_copy(
            update={"preferred_clusters": ("one",), "forbidden_clusters": ("two",)}
        ),
    )
    assert "load_modules" not in dispatch.model_dump()
    restored = DispatchSpec.model_validate_json(dispatch.model_dump_json())
    assert restored.job.jobs[0].config()().value == 3
    assert "request" not in restored.model_dump()
    assert restored.job.preferred_clusters == ("one",)
    assert restored.job.forbidden_clusters == ("two",)


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_submit_preserves_request_and_respects_dry_run(tmp_path, monkeypatch, dry_run, fresh):
    path = tmp_path / "job.yaml"
    save_configuration(load_configuration("tests/cli/first"), path)
    module = importlib.import_module("theseus.cli.submit")
    allocated = HardwareResult(
        chip=None,
        hosts=[
            ClusterMachine(
                name="host",
                cluster=Cluster(
                    name="one",
                    root="/root",
                    work="/work",
                    mount="rediss://user:secret@example.invalid",
                ),
                resources={},
                uv_groups=["cpu"],
                env={"KEEP": "existing", "XLA_FLAGS": "old"},
            )
        ],
        total_chips=0,
    )
    resolver = Mock(return_value=SolveResult(allocated, None))
    launch = Mock(return_value=Mock(ok=True, job_ids=("123",), logs=()))
    monkeypatch.setattr(module, "solve", resolver)
    monkeypatch.setattr(DispatchConfig, "load", Mock(return_value=DispatchConfig()))
    monkeypatch.setattr(DispatchSpec, "launch", launch)
    result = CliRunner().invoke(
        app,
        [
            "submit",
            "trial",
            str(path),
            "cli.value=13",
            "--cluster",
            "one",
            "--extras",
            "huggingface",
            "--extras",
            "cpu",
            "--env",
            "XLA_FLAGS=old override",
            "--env",
            "XLA_FLAGS=--xla_gpu_enable_command_buffer= --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER",
            "--env",
            "EMPTY=",
            *(["--dry-run"] if dry_run else []),
            *(["--fresh"] if fresh else []),
            "-n",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert resolver.call_args.args[0].preferred_clusters == ("one",)
    assert resolver.call_args.args[0].config.cli.value == 13
    assert resolver.call_args.args[0].gpus == 0
    assert allocated.hosts[0].env == {
        "KEEP": "existing",
        "EMPTY": "",
        "XLA_FLAGS": "--xla_gpu_enable_command_buffer= --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER",
    }
    assert allocated.hosts[0].cluster.mount == "rediss://user:secret@example.invalid"
    assert allocated.hosts[0].uv_groups == ["cpu", "huggingface"]
    if dry_run:
        payload = json.loads(result.stdout)
        assert (payload["nonce"] == OmegaConf.load(path).execution.nonce) is not fresh
        assert payload["nonce"] == payload["job"]["nonce"]
        assert '"preferred_clusters"' in result.output
        assert '"request"' not in result.output
        assert '"KEEP": "<redacted>"' in result.output
        assert '"XLA_FLAGS": "<redacted>"' in result.output
        assert '"mount": "<redacted>"' in result.output
        assert "existing" not in result.output
        assert "rediss://user:secret@example.invalid" not in result.output
        assert "--xla_gpu_enable_command_buffer" not in result.output
        launch.assert_not_called()
    else:
        launch.assert_called_once()
        assert "SUBMIT | submitted trial: 123" in result.stderr
        assert result.stdout == ""


@pytest.mark.parametrize(
    "assignment", ["MISSING_VALUE", "=value", "BAD-NAME=value", "9NAME=value"]
)
def test_submit_rejects_invalid_environment_before_solving(
    tmp_path, monkeypatch, assignment
):
    path = tmp_path / "job.yaml"
    path.write_text("")
    module = importlib.import_module("theseus.cli.submit")
    resolver = Mock()
    monkeypatch.setattr(module, "solve", resolver)
    result = CliRunner().invoke(
        app, ["submit", "trial", str(path), "--env", assignment]
    )
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "constraints,expected",
    [
        ({"preferred_clusters": ["two"]}, ["second"]),
        ({"forbidden_clusters": ["one"]}, ["second"]),
        ({"preferred_clusters": ["one"], "forbidden_clusters": ["one"]}, None),
        ({"preferred_clusters": ["typo"]}, None),
    ],
)
def test_solver_applies_request_before_contacting_providers(
    monkeypatch, constraints, expected
):
    inventory = DispatchConfig(
        clusters={
            name: ClusterConfig(root="/root", work="/work") for name in ("one", "two")
        },
        hosts={
            "first": SSHConfig(ssh="first", cluster="one"),
            "second": SSHConfig(ssh="second", cluster="two"),
        },
    )
    calls = []

    def available(self, execution, config, timeout):
        calls.append(self.name)
        return None

    monkeypatch.setattr(SSHProvider, "solve", available)
    if expected is None:
        with pytest.raises(ValueError):
            solve(
                Combobulator()
                .run(CLIJob)
                .model_copy(
                    update={key: tuple(value) for key, value in constraints.items()}
                ),
                inventory,
            )
        assert not calls
    else:
        solve(
            Combobulator()
            .run(CLIJob)
            .model_copy(
                update={key: tuple(value) for key, value in constraints.items()}
            ),
            inventory,
        )
        assert calls == expected


def test_editor_preserves_fields_and_lineage_across_edits():
    editor = Configurator(
        load_configuration("tests/cli/first", overrides=["cli.value=19"])
    )
    editor.ask = Mock(side_effect=["tests/cli/second", "tests/cli/second", "Fresh"])
    edit_steps(editor, "Add job")
    assert editor.document.config.cli.value == 19
    assert editor.document.config.other.value == 7
    editor.ask = Mock(side_effect=["cli.value", "config.cli.value = 19", "23"])
    edit_fields(editor)
    editor.ask = Mock(side_effect=["sharding.tp", "execution.sharding.tp = 1", "2"])
    edit_fields(editor, resources=True)
    assert editor.document.execution.sharding.tp == 2
    editor.ask = Mock(side_effect=[
        "activation_checkpointing", "execution.sharding.activation_checkpointing = False", "true"
    ])
    edit_fields(editor, resources=True)
    assert Combobulation.deserialize(editor.document).sharding.activation_checkpointing is True
    editor.ask = Mock(return_value="2. tests/cli/second")
    edit_steps(editor, "Remove step")
    assert "other" not in editor.document.config
    assert editor.document.config.cli.value == 23
    editor.ask = Mock(
        side_effect=[
            "tests/cli/second",
            "tests/cli/second",
            "Branch",
            "Previous step",
        ]
    )
    edit_steps(editor, "Add job")
    editor.ask = Mock(return_value="1. tests/cli/first")
    with pytest.raises(ValueError, match="depends"):
        edit_steps(editor, "Remove step")
    assert len(editor.document.execution.steps) == 2


@pytest.mark.parametrize("existing", [None, "replace", "keep"])
def test_rich_prompts_save_a_real_document(tmp_path, existing):
    path = tmp_path / "interactive.yaml"
    if existing:
        path.write_text("original content")
    answers = ["1", "tests/cli/first", "1", "7", str(path)]
    if existing:
        answers.extend(["1", "8"] if existing == "keep" else ["2"])
    result = CliRunner().invoke(app, ["configure"], input="\n".join([*answers, ""]))
    assert result.exit_code == 0, result.output
    assert "How should this job start?" not in result.output
    assert "[1. tests/cli/first] (fresh)" in result.output
    if existing == "keep":
        assert path.read_text() == "original content"
    else:
        restored = Combobulation.deserialize(OmegaConf.load(path))
        assert restored.jobs == (CLIJob,)
        assert restored.config.cli.value == 3


@pytest.mark.parametrize("responses", ["8\n", "", "invalid\n8\n"])
def test_rich_prompts_quit_without_saving(tmp_path, monkeypatch, responses):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["configure"], input=responses)
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize(
    "answers",
    [
        ["4", "cli.value", "1", ":cancel"],
        ["1", "tests/cli/second", "1", ":cancel"],
        ["5", "sharding.tp", "1", ":cancel"],
    ],
)
def test_cancel_edit_preserves_previous_changes(tmp_path, answers):
    path = tmp_path / "cancelled.yaml"
    # First finish an edit, then cancel a different operation before saving.
    responses = [
        "1",
        "tests/cli/first",
        "1",
        "4",
        "cli.value",
        "1",
        "17",
        *answers,
        "7",
        str(path),
        "",
    ]
    result = CliRunner().invoke(app, ["configure"], input="\n".join(responses))
    assert result.exit_code == 0, result.output
    assert "Edit cancelled." in result.output
    saved = OmegaConf.load(path)
    execution = Combobulation.deserialize(saved)
    assert execution.jobs == (CLIJob,)
    assert execution.config.cli.value == 17
    assert saved.execution.sharding.tp == 1
