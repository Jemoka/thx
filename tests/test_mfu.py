"""MFU follows completed updates and reaches the same node store as loss."""
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from theseus.base import ExecutionSpec, TheoreticalFLOPS
from theseus.config import build, configuration
from theseus.store import RecordStore
from tests.test_stack_e2e import TinyModel, TinyTrainer


class CountedModel(TinyModel):
    def flops(self, seq: int) -> float:
        return float(6 * seq * 4 * 4)


class CountedTrainer(TinyTrainer):
    MODEL = CountedModel

    def batch(self, slice="train"):
        return jax.tree.map(lambda a: np.repeat(a, 2, axis=0), super().batch(slice))


@pytest.mark.parametrize("completed_steps", [0, 1])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("known_peak", [True, False])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_mfu_reporting_preserves_jit(tmp_path, monkeypatch, accumulation, known_peak, completed_steps, activation_checkpointing):
    cfg = build(*CountedTrainer.config())
    cfg.architecture.dtype.activation = "float32"
    cfg.training.batch_size = 2
    cfg.training.per_device_batch_size = 2 // accumulation
    cfg.training.tokens = 32  # Four updates of two four-token sequences.
    cfg.training.evaluate = False
    cfg.training.analyze = False
    cfg.logging.report_interval = 2
    cfg.logging.validation_interval = 100
    cfg.logging.checkpoint_interval = 100
    spec = ExecutionSpec.local(str(tmp_path), name="mfu")
    spec.topology.shard = spec.topology.shard.model_copy(update={
        "activation_checkpointing": activation_checkpointing,
        "fsdp": activation_checkpointing,
    })
    spec.topology.chip = spec.topology.chip.model_copy(update={
        "flops": TheoreticalFLOPS(float32=1.0 if known_peak else None),
    })
    monkeypatch.setattr("theseus.training.base.Profiler", Mock())
    clock = Mock(side_effect=[100.0, 120.0, 125.0])
    monkeypatch.setattr("theseus.training.base.time", SimpleNamespace(perf_counter=clock))
    with configuration(cfg):
        trainer = CountedTrainer(spec)
        trainer.setup()
        trainer.state = trainer.state.replace(step=jnp.asarray(completed_steps, dtype=jnp.int32))
        train_step = Mock(wraps=trainer._make_train_step())
        trainer._make_train_step = Mock(return_value=train_step)
        params = sum(p.size for p in jax.tree.leaves(trainer.state.params))
        trainer.run()
        trainer.finish()

    train_step.lower.assert_not_called()
    assert train_step.call_count == 4 - completed_steps
    assert clock.call_count == 3
    assert int(trainer.state.step) == 4
    store = RecordStore(spec.hardware)
    try:
        first = store.query().node(trainer.node.model_copy(update={"seq": 1 - completed_steps})).select(raw=True)[0]
        second = store.query().node(trainer.node.model_copy(update={"seq": 3 - completed_steps})).select(raw=True)[0]
        assert first["train/tokens_per_second_per_device"] == pytest.approx(8 * (2 - completed_steps) / 20 / spec.topology.device_count)
        assert second["train/tokens_per_second_per_device"] == pytest.approx(16 / 5 / spec.topology.device_count)
        if known_peak:
            work = (6 * 4 * 4 * 4 * 2 + 20 * params) / 1e12
            assert first["train/mfu"] == pytest.approx(work * (2 - completed_steps) / 20)
            assert second["train/mfu"] == pytest.approx(work * 2 / 5)
        else:
            assert "train/mfu" not in first
            assert "train/mfu" not in second
    finally:
        store.close()
