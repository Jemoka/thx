"""Dispatch restoration uses checkpoint metadata and the merged runtime config."""

from dataclasses import dataclass
from unittest.mock import Mock

import comet_ml
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict

from theseus.config import build, configuration, current_config, field
from theseus.execute.combobulator import Combobulation, Combobulator
from theseus.execute.dispatch import DispatchSpec
from theseus.job import CometLoggingConfig, CometLoggingJob, RestoreableJob
from theseus.registry import job
from tests.test_execute_run import hardware, new_runner


@dataclass
class RestoreConfig:
    value: int = field("runner/value", default=1)
    fail_once: bool = field("runner/fail_once", default=False)
    fail_before_checkpoint: bool = field("runner/fail_before_checkpoint", default=False)


@job("tests/run/checkpoint")
class CheckpointRunJob(CometLoggingJob, RestoreableJob):
    observations = []

    @classmethod
    def config(cls):
        return [RestoreConfig, CometLoggingConfig]

    @property
    def template(self):
        return {"value": jax.ShapeDtypeStruct((), np.int32)}

    def initialize(self):
        self.state = {"value": jnp.asarray(17, dtype=jnp.int32)}

    def surgery(self, partial):
        return {"value": jnp.asarray(17, dtype=jnp.int32)}

    def apply(self, state, metadata):
        self.state = state

    def run(self):
        super().run()
        self.observations.append({
            "type": type(self), "spec": self.spec.model_copy(),
            "node": self.node.model_copy(), "base": self.base,
            "config": OmegaConf.to_container(current_config(), resolve=True),
            "value": self.args.value, "state": int(self.state["value"]),
        })
        if self.args.fail_before_checkpoint:
            raise RuntimeError("interrupted before checkpoint")
        if self.base is None:
            self.save(self.state, {})
        if self.args.fail_once and len(self.observations) == 1:
            raise RuntimeError("checkpoint saved before interruption")


@job("tests/run/inspect-checkpoint")
class InspectCheckpointJob(CheckpointRunJob):
    pass


@pytest.fixture(autouse=True)
def reset_restore_jobs(monkeypatch):
    CheckpointRunJob.observations.clear()
    monkeypatch.setattr(comet_ml, "start", Mock())
    monkeypatch.setattr("jax.experimental.multihost_utils.sync_global_devices", lambda _: None)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("override", [None, 9])
def test_dispatch_restores_job_spec_config_and_requested_class(hardware, resume, override):
    saved = build(*CheckpointRunJob.config())
    saved.runner.value = 5
    saved.logging.remote = True
    with open_dict(saved):
        saved.saved_only = {"marker": 29}
    seed_spec = new_runner(DispatchSpec(
        name="original", project="training", group="saved", hardware=hardware,
        job=Combobulator().run(CheckpointRunJob),
    )).spec
    seed_spec.id = "saved-job-id"
    with configuration(saved):
        seed = CheckpointRunJob(seed_spec)
        seed()
    checkpoint = seed.node.model_copy()
    CheckpointRunJob.observations.clear()
    comet_ml.start.reset_mock()

    execution = getattr(Combobulation(steps=()), "resume" if resume else "branch")(
        InspectCheckpointJob, base=lambda query: query.node(checkpoint).checkpoint()
    )
    # The full dispatch config is intentional: even its default value overrides
    # the saved value 5. Runtime CLI overrides are merged after those defaults.
    execution.config.logging.remote = True
    allocated = hardware.model_copy(update={"total_chips": 2})
    dispatch = DispatchSpec(
        name="inspection", project="analysis", group="requested", hardware=allocated,
        job=execution,
    )
    runtime = OmegaConf.create({"runner": {"value": override}}) if override is not None else None
    assert new_runner(dispatch, runtime).run()
    observed = CheckpointRunJob.observations[0]
    assert observed["type"] is InspectCheckpointJob
    expected_value = override if override is not None else 1
    assert observed["value"] == expected_value and observed["state"] == 17
    assert observed["config"]["saved_only"] == {"marker": 29}
    assert observed["config"]["runner"]["value"] == expected_value
    assert observed["base"] == checkpoint
    if resume:
        assert observed["node"] == checkpoint
    else:
        assert observed["node"].parent == checkpoint.serialize()
        assert observed["node"].nonce != checkpoint.nonce
    spec = observed["spec"]
    expected_identity = ("original", "training", "saved") if resume else ("inspection", "analysis", "requested")
    assert (spec.name, spec.project, spec.group) == expected_identity
    assert spec.hardware == allocated and spec.topology is None
    assert spec.id == ("saved-job-id" if resume else None)
    assert spec.execution_id == dispatch.nonce and spec.tag == "0:tests/run/inspect-checkpoint"
    experiment = comet_ml.start.return_value
    assert comet_ml.start.call_args.kwargs["experiment_key"] == comet_ml.get_experiment_key(observed["node"].nonce)
    experiment.set_name.assert_called_with(expected_identity[0])
    assert experiment.log_parameters.call_args.args[0] == observed["config"]


def test_checkpoint_retry_keeps_node_and_comet_identity(hardware):
    execution = Combobulator().run(CheckpointRunJob)
    execution.config.runner.fail_once = True
    execution.config.logging.remote = True
    dispatch = DispatchSpec(name="retry", project="tests", group="execute", hardware=hardware, job=execution)
    with pytest.raises(RuntimeError, match="checkpoint saved before interruption"):
        new_runner(dispatch).run()
    assert new_runner(dispatch).run()
    first, second = CheckpointRunJob.observations
    assert first["node"] == second["node"] == second["base"]
    assert first["spec"].execution_id == second["spec"].execution_id == dispatch.nonce
    keys = [call.kwargs["experiment_key"] for call in comet_ml.start.call_args_list]
    assert len(keys) == 2 and keys[0] == keys[1]
    assert new_runner(dispatch).run()
    assert len(CheckpointRunJob.observations) == 2


def test_retry_without_checkpoint_starts_fresh_node_and_comet(hardware):
    execution = Combobulator().run(CheckpointRunJob)
    execution.config.runner.fail_before_checkpoint = True
    execution.config.logging.remote = True
    dispatch = DispatchSpec(name="retry", project="tests", group="execute", hardware=hardware, job=execution)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="interrupted before checkpoint"):
            new_runner(dispatch).run()
    first, second = CheckpointRunJob.observations
    assert first["node"].nonce != second["node"].nonce
    assert first["base"] is None and second["base"] is None
    assert first["spec"].execution_id == second["spec"].execution_id == dispatch.nonce
    keys = [call.kwargs["experiment_key"] for call in comet_ml.start.call_args_list]
    assert len(keys) == 2 and keys[0] != keys[1]
