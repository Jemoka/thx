"""Native initialization, state transitions, and restoration for trainer variants."""

from contextlib import nullcontext
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from theseus.base import ExecutionSpec
from theseus.config import build, configuration
from theseus.data.datasets import DatasetComponent
from theseus.training.flywheel.strategy import Sampling
from theseus.training.contrastive import ContrastiveTrainer, ContrastiveTrainState
from theseus.training.kl_divergence import KLDivergenceTrainer, KLDivergenceTrainState
from theseus.training.lora import LoRATrainer, LoRATrainState
from tests.test_stack_e2e import TinyModel


class LocalData(DatasetComponent):
    DATASET_KEY = "fixture"


@pytest.mark.parametrize("trainer_type,state_type", [
    (ContrastiveTrainer, ContrastiveTrainState),
    (KLDivergenceTrainer, KLDivergenceTrainState),
    (LoRATrainer, LoRATrainState),
])
def test_specialized_trainer_checkpoint_lifecycle(tmp_path, trainer_type, state_type):
    class Trainer(trainer_type):
        MODEL = TinyModel
        DATASET = [Sampling(LocalData, 1, "padded")]
        EVALUATION = []

    data = tmp_path / "data" / "fixture"
    data.mkdir(parents=True)
    tokens = np.tile(np.arange(4, dtype=np.uint32), (8, 1))
    for split in ("train", "val"):
        tokens.tofile(data / f"{split}.bin")
        np.ones_like(tokens, dtype=np.bool_).tofile(data / f"{split}.bin.mask")
    (data / "shape.json").write_text(json.dumps({"train": [8, 4], "val": [8, 4]}))
    cfg = OmegaConf.merge(build(*Trainer.config()), {
        "architecture": {"block_size": 4, "dtype": {"param": "float32", "activation": "float32"}},
        "training": {"batch_size": 1, "per_device_batch_size": 1, "validation_steps": 1},
        "data": {"suffix": ""},
    })
    spec = ExecutionSpec.local(str(tmp_path), name="source")
    with configuration(cfg):
        trainer = Trainer(spec)
        assert not hasattr(trainer, "state")
        trainer.setup()
        trainer.state = jax.jit(lambda state: state.apply_gradients(
            grads=jax.tree.map(jnp.ones_like, state.params)
        ))(trainer.state)
        if trainer_type is LoRATrainer:
            trainer._transition_to_lora()
            assert int(trainer.state.step) == 1
        elif trainer_type is KLDivergenceTrainer:
            trainer._snapshot_reference()
            trainer.state = trainer.state.replace(beta=jnp.asarray(trainer.kl_config.beta, dtype=jnp.float32))
        assert isinstance(trainer.state, state_type)
        trainer.checkpoint()
        trainer.chkpt_manager.checkpointer.wait_until_finished()
        node = trainer.node.model_copy()
        expected = jax.device_get(trainer.state)
        trainer.finish()

        restored = Trainer(spec.model_copy(update={"name": "restored"}), base=node)
        try:
            restored.setup(resume=True)
            assert isinstance(restored.state, state_type)
            assert int(restored.state.step) == 1
            for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(restored.state), strict=True):
                np.testing.assert_array_equal(before, after)
        finally:
            restored.finish()


@pytest.mark.parametrize("evaluation", [False, True])
def test_backbone_initialization_uses_native_setup(tmp_path, monkeypatch, evaluation):
    from flax.core import freeze
    from theseus.training.backbone import BackbonedTrainer
    from theseus.experiments.benchmark import BackboneEvaluate

    base = BackboneEvaluate if evaluation else BackbonedTrainer

    class Owned(base):
        EVALUATION = []
        DATASET = []

        def _init_data(self, spec):
            pass

    cfg = build(*Owned.config())
    cfg.architecture.backbone.implementation = "llama"
    cfg.architecture.backbone.weights = "fixture"
    cfg.architecture.block_size = 4
    cfg.architecture.dtype.param = "float32"
    cfg.architecture.dtype.activation = "float32"
    cfg.training.per_device_batch_size = 1
    if not evaluation:
        cfg.training.batch_size = 1
    model = TinyModel(param_dtype="float32", activation_dtype="float32")
    params = freeze(jax.device_get(model.init(jax.random.PRNGKey(7), jnp.zeros((1, 4), dtype=jnp.int32))["params"]))
    module = "theseus.experiments.benchmark" if evaluation else "theseus.training.backbone"
    monkeypatch.setattr(f"{module}.load_backbone", lambda: (model, params))
    with configuration(cfg):
        job = Owned(ExecutionSpec.local(str(tmp_path)))
        assert not hasattr(job, "state")
        try:
            job.setup()
            initial = job.state
            job.setup()
            assert job.state is initial
            for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(job.state.params), strict=True):
                np.testing.assert_array_equal(expected, actual)
        finally:
            job.finish()




def test_lora_training_crosses_phase_boundary_with_cached_batches(tmp_path):
    from theseus.inference.base import InferenceJob

    class Trainer(LoRATrainer):
        MODEL = TinyModel
        DATASET = [Sampling(LocalData, 1, "padded")]
        EVALUATION = []

    data = tmp_path / "data" / "fixture"
    data.mkdir(parents=True)
    tokens = np.tile(np.arange(4, dtype=np.uint32), (8, 1))
    for split in ("train", "val"):
        tokens.tofile(data / f"{split}.bin")
        np.ones_like(tokens, bool).tofile(data / f"{split}.bin.mask")
    (data / "shape.json").write_text(json.dumps({"train": [8, 4], "val": [8, 4]}))
    cfg = OmegaConf.merge(build(*Trainer.config()), {
        "architecture": {"block_size": 4, "dtype": {"param": "float32", "activation": "float32"}},
        "training": {"batch_size": 1, "per_device_batch_size": 1, "validation_steps": 1,
                     "pre_lora_tokens": [4], "post_lora_tokens": [4], "validation": True, "evaluate": False},
        "logging": {"validation_interval": 100, "checkpoint_interval": 100},
        "data": {"suffix": ""},
    })
    with configuration(cfg):
        trainer = Trainer(ExecutionSpec.local(str(tmp_path)))
        try:
            trainer.setup()
            first = trainer.batch()
            assert trainer.batch() is first
            assert not trainer._in_lora_phase
            trainer.train()
            assert trainer._in_lora_phase
            assert isinstance(trainer.state, LoRATrainState)
            assert int(trainer.state.step) >= 2
            assert any(np.any(value != 0) for value in jax.tree.leaves(trainer.state.params["lora_B"]))
            batch = jax.tree.map(jnp.asarray, trainer.batch())
            logits, _, _ = InferenceJob.forward(trainer.state, trainer.state.params,
                                               (batch["x"], batch["y"], batch["padding_mask"]), deterministic=True)
            expected, _, _ = trainer.forward(trainer.state, trainer.state.params, batch, deterministic=True)
            np.testing.assert_allclose(logits, expected)
        finally:
            trainer.finish()




@pytest.mark.parametrize("accumulation", [1, 2, 4])
@pytest.mark.parametrize("completed_steps", [0, 2, 6])
def test_training_loop_counts_completed_updates(accumulation, completed_steps):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from theseus.training.base import BaseTrainer

    def update(state, batch, key, accumulate_steps):
        return SimpleNamespace(step=state.step + 1), jnp.array(1.), {}, jnp.array(0.)

    trainer = Mock(
        base=None,
        state=SimpleNamespace(step=completed_steps),
        args=SimpleNamespace(
            validate=True, analyze=False, evaluate=False,
            batch_size=8, block_size=128, report_interval=2,
            validation_interval=6, checkpoint_interval=8,
        ),
        total_steps=6,
        total_batches=6 * accumulation,
        accumulate_steps=accumulation,
        dropout_key=jax.random.PRNGKey(0),
    )
    trainer._profiler.measure_step.side_effect = nullcontext
    step = Mock(side_effect=update)
    validate = Mock(return_value=(None, {}))
    trainer._make_train_step.return_value = step
    trainer._make_valid_step.return_value = validate
    trainer.main_process.return_value = True
    trainer.schedule.return_value = 0.
    checkpoints = []
    trainer.checkpoint.side_effect = lambda: checkpoints.append(
        trainer.state.step
    )

    BaseTrainer.train(trainer)

    assert trainer.state.step == 6
    assert step.call_count == trainer.tick.call_count == 6 - completed_steps
    assert trainer._profiler.measure_step.call_count == step.call_count
    assert [call.kwargs['step'] for call in validate.call_args_list] == [
        step for step in (2, 6) if step > completed_steps
    ]
    assert checkpoints == [
        step for step in (2, 4, 6) if step > completed_steps
    ]
    assert [call.args[0]['train/tokens'] for call in trainer.log.call_args_list] == [
        step * 8 * 128
        for step in (2, 2, 4, 6, 6)
        if step > completed_steps
    ]
