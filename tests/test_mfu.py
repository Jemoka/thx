"""MFU follows completed updates and reaches the same node store as loss."""
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock

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
    cfg.logging.remote = False
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
    monkeypatch.setattr("theseus.training.base.Profiler", MagicMock())
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
        step_flops = 6 * 4 * 4 * 4 * 2 + 20 * params
        assert first["train/flops"] == step_flops * 2
        assert second["train/flops"] == step_flops * 4
        if known_peak:
            work = (6 * 4 * 4 * 4 * 2 + 20 * params) / 1e12
            assert first["train/mfu"] == pytest.approx(work * (2 - completed_steps) / 20)
            assert second["train/mfu"] == pytest.approx(work * 2 / 5)
        else:
            assert "train/mfu" not in first
            assert "train/mfu" not in second
    finally:
        store.close()


def test_flops_restore_preserves_old_cost_and_terminal_report(tmp_path, monkeypatch):
    cfg = build(*CountedTrainer.config())
    cfg.training.batch_size = 2
    cfg.training.per_device_batch_size = 1
    cfg.training.tokens = 16
    cfg.training.evaluate = False
    cfg.training.analyze = False
    cfg.logging.report_interval = 100  # Only terminal validation logs counters.
    cfg.logging.validation_interval = 100
    cfg.logging.checkpoint_interval = 100
    cfg.logging.remote = False
    monkeypatch.setattr("theseus.training.base.Profiler", MagicMock())
    spec = ExecutionSpec.local(str(tmp_path), name="flops")
    with configuration(cfg):
        first = CountedTrainer(spec)
        first.setup()
        first.run()
        cost = first._step_flops
        node = first.node.model_copy(update={"seq": first.node.seq - 1}, deep=True)
        first.finish()
    assert cost is not None
    cfg.training.tokens = 24
    # A new configuration has a different cost; old work must not be repriced.
    monkeypatch.setattr(CountedModel, "flops", lambda self, seq: float(12 * seq * 4 * 4))
    with configuration(cfg):
        second = CountedTrainer(spec, base=node)
        second.setup(resume=True)
        assert second._train_flops == 2 * cost
        second.run()
        expected = 2 * cost + second._step_flops
        assert second._train_flops == expected
        final_node = second.node.model_copy(update={"seq": second.node.seq - 1}, deep=True)
        second.finish()
    store = RecordStore(spec.hardware)
    try:
        row = store.query().node(final_node).select(raw=True)[0]
        assert row["train/tokens"] == 24
        assert row["train/flops"] == expected
    finally:
        store.close()


@pytest.mark.parametrize("trainer_class", [TinyTrainer, CountedTrainer])
def test_unknown_flops_and_historical_restore_use_warned_estimates(tmp_path, monkeypatch, trainer_class):
    cfg = build(*trainer_class.config())
    cfg.training.evaluate = False
    cfg.training.analyze = False
    cfg.logging.remote = False
    warning = Mock()
    monkeypatch.setattr("theseus.training.base.logger.warning", warning)
    with configuration(cfg):
        trainer = trainer_class(ExecutionSpec.local(str(tmp_path), name="unknown"))
        trainer.setup()
        trainer.apply(trainer.state.replace(step=jnp.asarray(5)), {})
        trainer.mfu()
        params = sum(p.size for p in jax.tree.leaves(trainer.state.params))
        model_flops = 6 * (params if trainer_class is TinyTrainer else 16) * trainer.args.block_size
        expected = model_flops * trainer.args.batch_size + 20 * params
        assert trainer._step_flops == expected
        assert trainer._train_flops == 5 * expected
        assert warning.call_count == (2 if trainer_class is TinyTrainer else 1)
        trainer._train_flops += expected
        trainer.mfu()
        assert trainer._train_flops == 6 * expected  # Refresh cannot reprice history.
        trainer.finish()
