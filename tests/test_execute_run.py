from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from theseus.base import ExecutionSpec, Node, SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.config import field
from theseus.execute.combobulator import Combobulator
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.run import Runner
from theseus.execute.run.run import main
from theseus.job import BasicJob
from theseus.registry import job
from theseus.store import ObjectStore, ValueRow


@dataclass
class RunConfig:
    value: int = field("runner/value", default=1)


class RecordingJob(BasicJob[RunConfig]):
    calls: list[tuple[str, Node | None, bool, int]] = []

    @classmethod
    def config(cls) -> type[RunConfig]:
        return RunConfig

    def state_restore(self, state: ValueRow) -> None:
        self.restored = state

    def state_init(self) -> None:
        self.restored = None

    def __call__(self, resume: bool = False) -> None:
        self.requested_resume = resume
        super().__call__(resume)

    def run(self) -> None:
        RecordingJob.calls.append(
            (self.spec.tag or "", self.base, self.requested_resume, self.args.value)
        )
        self.store.value(
            self.node,
            {"_x_checkpoint": True, "runner/value": self.args.value},
        )


class UnregisteredRunJob(RecordingJob):
    pass


def new_runner(
    dispatch: DispatchSpec,
    overrides: DictConfig | None = None,
) -> Runner:
    spec = ExecutionSpec(
        name=dispatch.name,
        project=dispatch.project,
        group=dispatch.group,
        hardware=dispatch.hardware,
        topology=None,
        distributed=False,
        execution_id=dispatch.nonce,
    )
    return Runner.new(dispatch, spec, overrides)


@job("tests/run/first")
class FirstRunJob(RecordingJob):
    pass


@job("tests/run/second")
class SecondRunJob(RecordingJob):
    pass


@job("tests/run/failing")
class FailingRunJob(RecordingJob):
    attempts = 0
    finalizations = 0

    def run(self) -> None:
        type(self).attempts += 1
        raise RuntimeError("run failed")

    def finish(self) -> None:
        type(self).finalizations += 1
        super().finish()


@job("tests/run/interrupting")
class InterruptingRunJob(RecordingJob):
    def run(self) -> None:
        super().run()
        if not self.requested_resume:
            raise RuntimeError("run interrupted")


@pytest.fixture
def hardware(tmp_path: Path) -> HardwareResult:
    root = tmp_path / "root"
    root.mkdir()
    chip = SUPPORTED_CHIPS["cpu"]
    cluster = Cluster(
        name="test",
        root=str(root),
        work=str(tmp_path / "work"),
        objects=str(tmp_path / "objects"),
    )
    host = ClusterMachine(name="localhost", cluster=cluster, resources={chip: 1})
    return HardwareResult(chip=chip, hosts=[host], total_chips=1)


@pytest.fixture(autouse=True)
def reset_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingJob.calls.clear()
    FailingRunJob.attempts = 0
    FailingRunJob.finalizations = 0
    monkeypatch.setattr(
        "jax.experimental.multihost_utils.sync_global_devices", lambda _: None
    )


def test_dispatch_spec_assigns_and_round_trips_a_nonce(
    hardware: HardwareResult,
) -> None:
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=Combobulator().run(FirstRunJob),
    )

    restored = DispatchSpec.model_validate_json(dispatch.model_dump_json())
    invalid = dispatch.model_dump()
    invalid["nonce"] = None

    assert len(dispatch.nonce) == 6
    assert restored.nonce == dispatch.nonce
    with pytest.raises(ValidationError):
        DispatchSpec.model_validate(invalid)


def test_runner_redispatches_a_cloudpickled_job_idempotently(
    hardware: HardwareResult,
) -> None:
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=Combobulator().run(UnregisteredRunJob),
    )
    serialized = dispatch.model_dump_json()
    job_name = f"{UnregisteredRunJob.__module__}.{UnregisteredRunJob.__qualname__}"
    tag = f"0:{job_name}"

    restored = DispatchSpec.model_validate_json(serialized)
    assert restored.job.steps[0].job_name == job_name
    assert new_runner(restored).run()

    store = ObjectStore(hardware)
    try:
        finished = (
            store.query().execution(dispatch.nonce).tag(tag).finished().latest().all()
        )
    finally:
        store.close()

    assert finished
    assert RecordingJob.calls == [(tag, None, False, 1)]
    assert new_runner(DispatchSpec.model_validate_json(serialized)).run()
    # Loading by value restores the saved (empty) class-level call list.
    assert RecordingJob.calls == []


def test_runner_round_trips_config_and_uses_previous_checkpoint(
    hardware: HardwareResult,
) -> None:
    execution = Combobulator().run(FirstRunJob).branch(SecondRunJob)
    execution.config.runner.value = 3
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        nonce="execution-id",
        hardware=hardware,
        job=execution,
    )
    restored = DispatchSpec.model_validate_json(dispatch.model_dump_json())

    completed = new_runner(
        restored,
        OmegaConf.create({"runner": {"value": 7}}),
    ).run()

    assert completed
    assert RecordingJob.calls[0] == ("0:tests/run/first", None, False, 7)
    _, second_base, second_resume, second_value = RecordingJob.calls[1]
    assert second_base is not None
    assert second_base.name == "tests.execute.run"
    assert second_resume is False
    assert second_value == 7


def test_runner_stops_when_explicit_base_matches_no_checkpoint(
    hardware: HardwareResult,
) -> None:
    execution = Combobulator().run(
        FirstRunJob,
        base=lambda query: query.where("runner/value", ">", 100),
    )
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=execution,
    )

    assert new_runner(dispatch).run() is False
    assert RecordingJob.calls == []


def test_runner_uses_a_checkpoint_matching_an_explicit_base(
    hardware: HardwareResult,
) -> None:
    checkpoint = Node(name="tests.seed.run")
    store = ObjectStore(hardware)
    store.value(checkpoint, {"_x_checkpoint": True, "runner/value": 5})
    store.close()
    execution = Combobulator().run(
        FirstRunJob,
        base=lambda query: query.where("runner/value", "=", 5),
    )
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=execution,
    )

    restored = DispatchSpec.model_validate_json(dispatch.model_dump_json())

    assert new_runner(restored).run()
    assert RecordingJob.calls == [
        ("0:tests/run/first", checkpoint, False, 1),
    ]


def test_runner_skips_a_step_with_only_a_finish_tombstone(
    hardware: HardwareResult,
) -> None:
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=Combobulator().run(FirstRunJob),
    )
    store = ObjectStore(hardware, dispatch.nonce, "0:tests/run/first")
    store.value(Node(name="tests.execute.run"), {"_x_finished": True})
    store.close()

    assert new_runner(dispatch).run()
    assert RecordingJob.calls == []


@pytest.mark.parametrize(("operation", "resume"), [("branch", False), ("resume", True)])
def test_redispatch_skips_a_finished_prefix_and_starts_its_successor(
    hardware: HardwareResult,
    operation: str,
    resume: bool,
) -> None:
    execution = Combobulator().run(FirstRunJob)
    execution = getattr(execution, operation)(SecondRunJob)
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=execution,
    )
    checkpoint = Node(name="tests.execute.run")
    store = ObjectStore(hardware, dispatch.nonce, "0:tests/run/first")
    store.value(checkpoint, {"_x_checkpoint": True, "_x_finished": True})
    store.close()

    assert new_runner(dispatch).run()

    assert RecordingJob.calls == [
        ("1:tests/run/second", checkpoint, resume, 1),
    ]


def test_runner_finalizes_a_failed_job(hardware: HardwareResult) -> None:
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=Combobulator().run(FailingRunJob),
    )

    with pytest.raises(RuntimeError, match="run failed"):
        new_runner(dispatch).run()
    with pytest.raises(RuntimeError, match="run failed"):
        new_runner(dispatch).run()

    assert FailingRunJob.attempts == 2
    assert FailingRunJob.finalizations == 2


def test_redispatch_starts_continues_finishes_and_then_does_nothing(
    hardware: HardwareResult,
) -> None:
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=Combobulator().run(FirstRunJob).branch(InterruptingRunJob),
    )
    serialized = dispatch.model_dump_json()

    with pytest.raises(RuntimeError, match="run interrupted"):
        new_runner(DispatchSpec.model_validate_json(serialized)).run()

    store = ObjectStore(hardware)
    try:
        first_finished = (
            store.query()
            .execution(dispatch.nonce)
            .tag("0:tests/run/first")
            .finished()
            .all()
        )
        second_finished = (
            store.query()
            .execution(dispatch.nonce)
            .tag("1:tests/run/interrupting")
            .finished()
            .all()
        )
        second_checkpoints = (
            store.query()
            .execution(dispatch.nonce)
            .tag("1:tests/run/interrupting")
            .checkpoint()
            .latest()
            .all()
        )
    finally:
        store.close()

    assert first_finished
    assert second_finished == []
    assert len(second_checkpoints) == 1
    assert len(RecordingJob.calls) == 2

    assert new_runner(DispatchSpec.model_validate_json(serialized)).run()
    assert len(RecordingJob.calls) == 1
    resumed_tag, resumed_base, resumed, _ = RecordingJob.calls[-1]
    assert resumed_tag == "1:tests/run/interrupting"
    assert resumed_base == second_checkpoints[0]
    assert resumed is True

    store = ObjectStore(hardware)
    try:
        assert (
            store.query()
            .execution(dispatch.nonce)
            .tag("1:tests/run/interrupting")
            .finished()
            .all()
        )
    finally:
        store.close()

    assert new_runner(DispatchSpec.model_validate_json(serialized)).run()
    assert RecordingJob.calls == []

    fresh = DispatchSpec.model_validate_json(serialized).model_copy(
        update={"nonce": "fresh1"}
    )
    with pytest.raises(RuntimeError, match="run interrupted"):
        new_runner(fresh).run()
    assert len(RecordingJob.calls) == 2
    assert RecordingJob.calls[-2][1] is None
    assert RecordingJob.calls[-2][2] is False


def test_main_loads_json_applies_overrides_and_redispatches_idempotently(
    hardware: HardwareResult,
    tmp_path: Path,
) -> None:
    dispatch = DispatchSpec(
        name="run",
        project="tests",
        group="execute",
        hardware=hardware,
        job=Combobulator().run(FirstRunJob),
    )
    path = tmp_path / "dispatch.json"
    path.write_text(dispatch.model_dump_json())

    main([str(path), "runner.value=11"])
    assert RecordingJob.calls == [("0:tests/run/first", None, False, 11)]
    main([str(path), "runner.value=11"])

    assert RecordingJob.calls == []


@pytest.mark.parametrize("path_as_string", [False, True])
def test_dispatch_runs_locally_without_resolved_hardware(tmp_path, path_as_string):
    RecordingJob.calls.clear()
    dispatch = DispatchSpec(
        name="local",
        project="test",
        group="run",
        job=Combobulator().run(FirstRunJob).branch(SecondRunJob),
    )
    restored = DispatchSpec.model_validate_json(dispatch.model_dump_json())
    assert restored.hardware is None
    root = str(tmp_path) if path_as_string else tmp_path
    assert restored.run(root)
    assert len(RecordingJob.calls) == 2
    assert RecordingJob.calls[1][1] is not None
    # The same dispatch identity skips steps that already finished.
    assert restored.run(root)
    assert len(RecordingJob.calls) == 2
    assert restored.hardware is None


def test_dispatch_remote_launch_requires_hardware():
    from theseus.execute.config import DispatchConfig

    dispatch = DispatchSpec(
        name="missing",
        project="test",
        group="run",
        job=Combobulator().run(FirstRunJob),
    )
    with pytest.raises(ValueError, match="hardware must contain at least one host"):
        dispatch.launch(DispatchConfig())
