from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

import comet_ml
import numpy as np
import pytest
from loguru import logger
from omegaconf import OmegaConf

from theseus.base import Node
from theseus.config import build, configuration
from theseus.job import CometLoggingJob, LoggingJob, RestoreableJob
from theseus.training.base import BaseTrainer
from scripts.smoke.gpt_comet_fixture import TinyGPT


@pytest.fixture
def trainer(monkeypatch):
    instance = object.__new__(TinyGPT)
    instance._setup_complete = True
    instance.node = Node(name="fixtures.laptop.test", nonce="abcdef", seq=7)
    instance.spec = Mock(project="fixtures", group="laptop")
    instance.spec.name = "tiny-gpt-comet"
    instance.store = Mock()
    instance.chkpt_manager = Mock()
    instance.main_process = Mock(return_value=True)
    instance.train = Mock()
    monkeypatch.setattr(comet_ml, "start", Mock())
    monkeypatch.setattr("theseus.job.logger.add", Mock(return_value=123))
    monkeypatch.setattr("theseus.job.logger.remove", Mock())
    return instance


@pytest.mark.parametrize(
    "remote,primary", [(False, True), (False, False), (True, False)]
)
def test_disabled_or_worker_keeps_local_logging(trainer, remote, primary):
    trainer.main_process.return_value = primary
    with configuration(OmegaConf.create({"logging": {"remote": remote}})):
        trainer.run()
        trainer.log({"loss": np.float32(2)})
        trainer.finish()
    comet_ml.start.assert_not_called()
    logger.add.assert_not_called()
    logger.remove.assert_not_called()
    trainer.train.assert_called_once()
    trainer.store.value.assert_called_once_with(trainer.node, {"loss": 2.0})
    trainer.store.close.assert_called_once()
    assert CometLoggingJob.artifact is LoggingJob.artifact


def test_trainer_schema_defaults_to_local_logging():
    cfg = build(*TinyGPT.config())
    assert cfg.logging.remote is False
    assert cfg.architecture.n_layers > 0
    assert cfg.training.batch_size > 0


def test_remote_lifecycle_parameters_tags_and_metric_steps(trainer):
    cfg = OmegaConf.create({"logging": {"remote": True}, "lr": 0.01, "copy": "${lr}"})
    with configuration(cfg):
        trainer.run()
        experiment = comet_ml.start.return_value
        trainer.log({"loss": np.float32(2)})
        trainer.tick()
        trainer.log({"loss": 1.5})
        trainer.finish()
        trainer.finish()
    comet_ml.start.assert_called_once_with(
        experiment_key=comet_ml.get_experiment_key("abcdef"),
        mode="get_or_create",
        project_name="fixtures",
        experiment_config=ANY,
    )
    assert (
        comet_ml.start.call_args.kwargs["experiment_config"].auto_output_logging
        == "simple"
    )
    experiment.set_name.assert_called_once_with("tiny-gpt-comet")
    experiment.add_tags.assert_called_once_with(["laptop"])
    experiment.log_parameters.assert_called_once_with(
        {"logging": {"remote": True}, "lr": 0.01, "copy": 0.01}
    )
    assert experiment.log_metrics.call_args_list == [
        call(
            {
                "loss": 2.0,
                "node": Node(name=trainer.node.name, nonce="abcdef", seq=7).serialize(),
            },
            step=7,
        ),
        call({"loss": 1.5, "node": trainer.node.serialize()}, step=8),
    ]
    experiment.log_text.assert_not_called()
    experiment.end.assert_called_once()
    logger.add.assert_not_called()
    logger.remove.assert_not_called()
    trainer.train.assert_called_once()


def test_resume_reuses_key_and_branch_changes_key(trainer):
    with configuration(OmegaConf.create({"logging": {"remote": True}})):
        for node in [
            trainer.node,
            trainer.node.next(),
            Node(name=trainer.node.name, nonce="branch"),
        ]:
            trainer.node = node
            trainer.run()
            trainer.finish()
    keys = [c.kwargs["experiment_key"] for c in comet_ml.start.call_args_list]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_comet_failure_still_closes_local_resources(trainer):
    trainer._comet = Mock()
    trainer._comet.end.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        trainer.finish()
    trainer.store.close.assert_called_once()
    trainer.chkpt_manager.close.assert_called_once()
    assert trainer._comet is None


def test_checkpoint_emits_metric_only_after_save(trainer):
    trainer._comet = Mock()
    trainer.state = SimpleNamespace(step=8)
    trainer.accumulate_steps = 1
    with patch.object(RestoreableJob, "save") as save:
        BaseTrainer.checkpoint(trainer)
        trainer._comet.log_metrics.assert_called_once_with(
            {"checkpoint": 1, "node": trainer.node.serialize()},
            step=7,
        )
        assert type(trainer._comet.log_metrics.call_args.args[0]["checkpoint"]) is int
        trainer._comet.reset_mock()
        save.side_effect = RuntimeError("save failed")
        with pytest.raises(RuntimeError, match="save failed"):
            BaseTrainer.checkpoint(trainer)
        trainer._comet.log_metrics.assert_not_called()
