from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from theseus.config import build, configuration, field, configure
from theseus.model.module import Module
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.optimizers import AdamW, Muon, Optimizer


class OptimizerModel(Module):
    @classmethod
    def components(cls):
        return []


class OptimizerTrainer(BaseTrainer):
    MODEL = OptimizerModel
    CONFIG = BaseTrainerConfig
    DATASET = []


@pytest.mark.parametrize("name, optimizer", [("adamw", AdamW), ("muon", Muon)])
def test_declared_optimizer_matches_legacy_updates(name, optimizer):
    class Trainer(OptimizerTrainer):
        OPTIMIZER = optimizer

    assert optimizer.config in Trainer.config()
    cfg = build(*Trainer.config())
    cfg.optimization.weight_decay = 0.03
    trainer = object.__new__(Trainer)
    trainer.schedule = optax.linear_schedule(0.01, 0.001, 3)
    with configuration(cfg):
        actual = trainer.optimizer
        assert trainer.optimizer is actual
        factory, schema = optimizer.optimizer, optimizer.config
        expected = factory(trainer.schedule, configure(schema))

    params = {
        "dense": jnp.array([[1.0, 0.5], [-0.2, 0.3]]),
        "embed": jnp.ones((3, 2)),
        "lm_head": jnp.ones((2, 3)),
        "bias": jnp.ones(2),
    }
    grads = jax.tree.map(lambda p: p * 0.1, params)
    state, expected_state = actual.init(params), expected.init(params)
    update = jax.jit(actual.update)
    for _ in range(3):
        updates, state = update(grads, state, params)
        expected_updates, expected_state = expected.update(
            grads, expected_state, params
        )
        for value, reference in zip(
            jax.tree.leaves(updates), jax.tree.leaves(expected_updates)
        ):
            np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-7)
            assert np.isfinite(value).all()
        params = optax.apply_updates(params, updates)


def test_default_optimizer_and_custom_config_override():
    assert OptimizerTrainer.OPTIMIZER is AdamW

    @dataclass
    class SGDConfig:
        multiplier: float = field("optimization/sgd/multiplier", default=1.0)

    def sgd(lr, cfg):
        return optax.chain(optax.scale(cfg.multiplier), optax.sgd(lr))

    class SGDTrainer(OptimizerTrainer):
        OPTIMIZER = Optimizer(SGDConfig, sgd)

    assert SGDConfig in SGDTrainer.config()
    assert AdamW.config not in SGDTrainer.config()
    cfg = build(*SGDTrainer.config())
    cfg.optimization.sgd.multiplier = 2.0
    trainer = object.__new__(SGDTrainer)
    trainer.schedule = optax.linear_schedule(0.1, 0.0, 2)
    with configuration(cfg):
        optimizer = trainer.optimizer
        assert trainer.optimizer is optimizer
    params = jnp.array([1.0])
    state = optimizer.init(params)
    for expected in (-0.2, -0.1, 0.0):
        updates, state = optimizer.update(jnp.ones(1), state, params)
        np.testing.assert_allclose(updates, [expected], atol=1e-7)
        params = optax.apply_updates(params, updates)


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (4, 4), (2, 8, 4)])
@pytest.mark.parametrize("annotated", [False, True])
def test_muon_factored_state_preserves_only_unreduced_axes(shape, annotated):
    import flax.linen as nn
    from theseus.training.optimizers.muon import scale_by_muon

    names = ("row", "column") if len(shape) == 2 else ("group", "row", "column")
    values = jax.random.normal(jax.random.PRNGKey(7), shape)
    params = {"kernel": nn.LogicallyPartitioned(values, names) if annotated else values}
    optimizer = scale_by_muon()
    state = optimizer.init(params)
    reference = optimizer.init({"kernel": values})
    reduced = -1 if shape[-2] >= shape[-1] else -2
    expected_shape = list(shape)
    expected_shape[reduced] = 1
    expected_names = list(names)
    expected_names[reduced] = None
    for _ in range(3):
        moment = state.second_moment["kernel"]
        if annotated:
            assert moment.names == tuple(expected_names)
            assert state.momentum["kernel"].names == names
            moment = moment.value
        else:
            assert not isinstance(moment, nn.LogicallyPartitioned)
        assert moment.shape == tuple(expected_shape)
        updates, state = optimizer.update(params, state)
        expected, reference = optimizer.update({"kernel": values}, reference)
        for actual, target in zip(
            jax.tree.leaves((updates, state)),
            jax.tree.leaves((expected, reference)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, target, rtol=1e-5, atol=1e-7)
