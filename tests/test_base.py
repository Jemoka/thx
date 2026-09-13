from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from theseus.base import Node
from theseus.job import RestoreableJob
from theseus.training.base import BaseTrainer


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("analyze", [False, True])
def test_evaluation_and_periodic_checkpoint_share_each_tick_once(analyze, resumed) -> None:
    trainer = cast(Any, object.__new__(BaseTrainer))
    trainer.args = SimpleNamespace(
        validate=False,
        evaluate=False,
        analyze=analyze,
        report_interval=999,
        checkpoint_interval=4,
        validation_interval=6,
        batch_size=1,
        block_size=1,
    )
    trainer.accumulate_steps = 1
    trainer.total_batches = 10
    trainer.total_steps = 10
    trainer.dropout_key = jax.random.PRNGKey(0)
    trainer.state = SimpleNamespace(step=0)
    trainer.node = Node(name="train", nonce="abcdef")
    trainer.base = trainer.node.model_copy() if resumed else None
    if resumed:
        trainer.base.parent = "ancestor"  # Parent is absent from the synchronized wire node.
    trainer._setup_complete = True
    trainer.store = Mock()
    trainer.inference = None
    trainer.main_process = Mock(return_value=True)
    trainer.batch = Mock(return_value={})
    trainer._reshape_batch = Mock(side_effect=lambda batch: batch)
    trainer._to_global = Mock(side_effect=lambda batch: batch)
    trainer._make_train_step = Mock(
        return_value=Mock(
            side_effect=lambda state, batch, key, accumulate: (
                SimpleNamespace(step=state.step + 1),
                jnp.array(1.0),
                {},
                jnp.array(0.0),
            )
        )
    )

    evaluation_ticks: list[int] = []
    checkpoint_ticks: list[int] = []
    ticked: list[int] = []
    analysis_ticks = []
    trainer.analyses = [
        SimpleNamespace(run=Mock(side_effect=lambda: analysis_ticks.append(("first", trainer.node.seq)))),
        SimpleNamespace(run=Mock(side_effect=lambda: analysis_ticks.append(("second", trainer.node.seq)))),
    ]
    trainer.log = Mock(
        side_effect=lambda values: evaluation_ticks.append(trainer.node.seq)
    )
    trainer.checkpoint = Mock(
        side_effect=lambda: checkpoint_ticks.append(trainer.node.seq)
    )

    def tick() -> None:
        ticked.append(trainer.node.seq)
        trainer.node = trainer.node.next()

    trainer.tick = Mock(side_effect=tick)

    with patch("theseus.training.base.Profiler") as profiler:
        trainer.train()
        trainer.train()  # Re-entering a completed loop must not tick the restore again.
    profiler.assert_called_once()
    profiler.return_value.start.assert_called_once_with()

    offset = int(resumed)
    assert evaluation_ticks == [step + offset for step in [1, 7, 9]]
    assert checkpoint_ticks == [step + offset for step in [1, 5, 7, 9]]
    assert set(evaluation_ticks) <= set(checkpoint_ticks)
    assert ticked == list(range(offset, 10 + offset))
    if resumed:
        trainer.store.value.assert_called_once()
        assert trainer.base.seq == 0
    else:
        trainer.store.value.assert_not_called()
    assert analysis_ticks == (
        [(name, step + offset) for step in [1, 7, 9] for name in ["first", "second"]]
        if analyze else []
    )


def test_checkpoint_saves_state_without_progress_metadata() -> None:
    trainer = cast(Any, object.__new__(BaseTrainer))
    trainer.state = SimpleNamespace(step=0)
    trainer.accumulate_steps = 2
    trainer.main_process = Mock(return_value=False)

    with patch.object(RestoreableJob, "save") as save:
        trainer.checkpoint()

    save.assert_called_once_with(
        trainer.state,
        {},
    )


def test_surgery_initializes_only_requested_leaves() -> None:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    trainer = SimpleNamespace(
        init_key=jax.random.PRNGKey(0),
        template={
            "existing": jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding),
            "new": jax.ShapeDtypeStruct((2,), jnp.float32, sharding=sharding),
        },
        _new_state=lambda key: {
            "existing": jax.random.normal(key, (2,)),
            "new": jax.random.uniform(key, (2,)),
        },
    )

    initialized = BaseTrainer.surgery(
        trainer,
        {"existing": False, "new": True},
    )

    assert initialized["existing"] is None
    assert isinstance(initialized["new"], jax.Array)
    assert initialized["new"].sharding == sharding


def test_dataset_must_be_declared_by_trainer():
    from theseus.model.models import GPT
    from theseus.training.base import BaseTrainerConfig

    class UndeclaredTrainer(BaseTrainer):
        MODEL = GPT
        CONFIG = BaseTrainerConfig

    assert not hasattr(BaseTrainer, "DATASET")
    with pytest.raises(AttributeError, match="DATASET"):
        UndeclaredTrainer.config()

    class ExplicitTrainer(UndeclaredTrainer):
        DATASET = []

    assert ExplicitTrainer.config()


def test_finish_stops_owned_profiler_once():
    trainer = object.__new__(BaseTrainer)
    owned = trainer._profiler = Mock()
    trainer.train_dl = Mock()
    trainer.val_dl = Mock()
    with patch.object(RestoreableJob, "finish"):
        trainer.finish()
        trainer.finish()
    owned.close.assert_called_once_with()
    assert trainer._profiler is None
