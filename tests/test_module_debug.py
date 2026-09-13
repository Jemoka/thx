from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from flax import linen as nn
from flax import errors
from flax.linen import module as linen_module
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from loguru import logger
import numpy as np
from omegaconf import OmegaConf
import pytest

from theseus.base import Axis
from theseus.config import configuration, configure, current_config, field
from theseus.model.debug import Debugger
from theseus.model.module import Module
from theseus.training.base import BaseTrainer


class CompactModule(Module):
    width: int = field("architecture/width", default=2)

    @nn.compact
    def __call__(self, x, *, scale=2):
        return nn.Dense(self.width)(x) * scale


class SetupModule(Module):
    def setup(self):
        self.child = CompactModule()

    def __call__(self, x):
        return self.some_compact(x)

    @nn.compact
    def some_compact(self, x):
        return self.child(x)


class RepeatedModule(Module):
    def setup(self):
        self.child = CompactModule()

    def __call__(self, x):
        return self.child(x) + self.child(x + 1)


@pytest.fixture
def experiment():
    model = SetupModule()
    x = jnp.ones((1, 3))
    with jax.disable_jit():
        variables = model.init(jax.random.key(0), x)
    config = OmegaConf.create({"architecture": {"width": 2}})
    trace = Mock(side_effect=lambda: model.apply(variables, x))
    return SimpleNamespace(
        model=model, x=x, variables=variables, config=config, trace=trace
    )


@pytest.mark.parametrize("path", ["child", ("child",)])
def test_compact_reuse_is_zero_copy_and_logs_paths(experiment, path):
    e = experiment
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    original = e.model.apply(e.variables, e.x)
    try:
        with Debugger(e.trace, e.config, path) as (layer, inputs):
            assert inputs.x is e.x
            assert inputs["x"] is e.x
            assert inputs.scale == 2
            assert "x" in dir(inputs)
            assert set(inputs) == {"x", "scale"}
            assert len(inputs) == 2
            kernel = layer.scope.variables()["params"]["Dense_0"]["kernel"]
            assert kernel is e.variables["params"]["child"]["Dense_0"]["kernel"]
            assert (
                kernel.sharding
                is e.variables["params"]["child"]["Dense_0"]["kernel"].sharding
            )
            result = nn.Dense(2)(inputs.x) * inputs.scale
            np.testing.assert_allclose(result, original, rtol=1e-5)
    finally:
        logger.remove(sink)
    assert any("child/Dense_0/kernel" in message for message in messages)
    e.trace.assert_called_once()
    np.testing.assert_array_equal(e.model.apply(e.variables, e.x), original)


def test_noncompact_scope_and_nested_compact_method(experiment):
    e = experiment
    with Debugger(e.trace, e.config, ()) as (layer, inputs):
        assert not layer._state.in_compact_method
        with pytest.raises(errors.AssignSubModuleError):
            nn.Dense(2)
        np.testing.assert_allclose(
            layer.some_compact(inputs.x), e.model.apply(e.variables, e.x)
        )
        assert not layer._state.in_compact_method
        np.testing.assert_allclose(
            layer.child(inputs.x), e.model.apply(e.variables, e.x)
        )


def test_direct_session_configuration_and_cleanup(experiment):
    e = experiment
    stack = list(linen_module._context.module_stack)
    outer = OmegaConf.create({"architecture": {"width": 9}})
    with configuration(outer), jax.disable_jit(False):
        layer, inputs = Debugger(e.trace, e.config, "child")
        assert current_config() is not e.config
        assert jax.config.jax_disable_jit
        # Named construction uses a captured child scope with hydrated config.
        layer.close()
        layer.close()
        assert current_config() is outer
        assert not jax.config.jax_disable_jit
        assert linen_module._context.module_stack == stack
        with pytest.raises(RuntimeError, match="closed"):
            layer(inputs.x)
        with pytest.raises(RuntimeError, match="closed"):
            _ = layer.width


def test_configured_child_uses_captured_parameters(experiment):
    e = experiment

    # A compact parent's original child has an explicit name.
    class Parent(Module):
        @nn.compact
        def __call__(self, x):
            return configure(CompactModule, name="child")(x)

    model = Parent()
    with configuration(e.config):
        variables = model.init(jax.random.key(0), e.x)
        expected = model.apply(variables, e.x)
    with Debugger(lambda: model.apply(variables, e.x), e.config, ()) as (layer, inputs):
        result = configure(CompactModule, name="child")(inputs.x)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mode", ["missing", "shape"])
def test_no_new_or_incompatible_parameters(experiment, mode):
    e = experiment
    with Debugger(e.trace, e.config, "child") as (layer, inputs):
        if mode == "missing":
            with pytest.raises(errors.ScopeParamNotFoundError):
                nn.Dense(2, name="unknown")(inputs.x)
        else:
            with pytest.raises(errors.ScopeParamShapeError):
                nn.Dense(7)(inputs.x)


@pytest.mark.parametrize("failure", ["trace", "path", "body"])
def test_errors_restore_all_contexts(experiment, failure):
    e = experiment
    before = current_config()
    stack = list(linen_module._context.module_stack)
    with jax.disable_jit(False):
        with pytest.raises((ValueError, LookupError)):
            if failure == "trace":
                Debugger(Mock(side_effect=ValueError("trace failed")), e.config, ())
            elif failure == "path":
                Debugger(e.trace, e.config, "chil")
            else:
                with Debugger(e.trace, e.config, "child"):
                    raise ValueError("body failed")
        assert not jax.config.jax_disable_jit
        assert current_config() is before
        assert linen_module._context.module_stack == stack
        with Debugger(e.trace, e.config, "child"):
            pass


def test_discovery_exact_paths_repeated_calls_and_overlaps(experiment):
    e = experiment
    model = RepeatedModule()
    variables = model.init(jax.random.key(0), e.x)
    trace = lambda: model.apply(variables, e.x)
    assert Debugger.find(trace, e.config, CompactModule) == [("child",)]
    with Debugger(trace, e.config, "child") as (layer, inputs):
        np.testing.assert_array_equal(inputs.x, e.x)
        with pytest.raises(RuntimeError, match="active debugger"):
            Debugger(e.trace, e.config, ())
        with pytest.raises(RuntimeError, match="active debugger"):
            Debugger.find(e.trace, e.config, Module)


def test_rng_counters_and_mutable_state_are_isolated():
    class RandomModule(Module):
        def setup(self):
            self.offset = jax.random.uniform(self.make_rng("dropout"), ())

        @nn.compact
        def __call__(self, x):
            count = self.variable("stats", "count", lambda: jnp.array(0))
            count.value += 1
            return x + self.offset + jax.random.uniform(self.make_rng("dropout"), ())

    model = RandomModule()
    x = jnp.ones((1,))
    rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
    variables = model.init(rngs, x)
    count = variables["stats"]["count"]
    trace = lambda: model.apply(variables, x, rngs=rngs, mutable=["stats"])
    expected, _ = trace()
    with Debugger(trace, OmegaConf.create({}), ()) as (layer, inputs):
        np.testing.assert_array_equal(layer(inputs.x), expected)
        assert variables["stats"]["count"] is count
        np.testing.assert_array_equal(variables["stats"]["count"], count)


def test_autodiff_capture_is_rejected_and_cleans_up(experiment):
    e = experiment
    trace = lambda: jax.grad(lambda x: e.model.apply(e.variables, x).sum())(e.x)
    with pytest.raises(RuntimeError, match="eager forward"):
        Debugger(trace, e.config, "child")
    with Debugger(e.trace, e.config, "child"):
        pass


def test_trainer_default_trace_and_override(experiment):
    e = experiment
    class Trainer(BaseTrainer):
        forward = staticmethod(Mock(side_effect=lambda state, params, batch, key: e.model.apply({"params": params}, batch["x"])))

    trainer = object.__new__(Trainer)
    trainer.model = e.model
    trainer._debug_config = e.config
    trainer.mesh = Mesh(
        np.array(jax.devices()).reshape(-1, 1), (Axis.BATCH, Axis.SHARD)
    )
    trainer.local_replicas = jax.local_device_count()
    trainer.per_device_batch_size = 2
    trainer.accumulate_steps = 3
    per_step = trainer.per_device_batch_size * trainer.local_replicas
    local = np.arange(3 * per_step * 3, dtype=np.float32).reshape(3 * per_step, 3)
    params = jax.device_put(e.variables["params"], NamedSharding(trainer.mesh, P()))
    trainer.state = SimpleNamespace(params=params)
    trainer.batch = Mock(return_value={"x": local})
    trainer.sharding_context = SimpleNamespace(mesh=trainer.mesh, parameter_fwdbwd_mapping=())
    state = trainer.state
    assert trainer.find(CompactModule) == [("child",)]
    with trainer.debug("child") as (layer, inputs):
        assert inputs.x is trainer.forward.call_args.args[2]["x"]
        assert inputs.x.shape == (per_step, 3)
        np.testing.assert_array_equal(inputs.x, local[:per_step])
        assert inputs.x.sharding.is_equivalent_to(
            NamedSharding(trainer.mesh, P(Axis.BATCH, None)), 2
        )
        assert trainer.forward.call_args.args[1] is params
        assert trainer.state is state
    assert trainer.batch.call_count == 2
    trainer.trace = lambda state, batch, key, **kw: e.trace()
    with trainer.debug("child"):
        pass
    assert trainer.batch.call_count == 3


def test_capture_never_deepcopies_device_arrays(experiment, monkeypatch):
    e = experiment

    class CachedModule(CompactModule):
        def setup(self):
            self.cached = e.x

    bound = CachedModule().bind({"params": e.variables["params"]["child"]})
    assert bound.cached is e.x
    trace = lambda: bound(e.x)
    with monkeypatch.context() as patches:
        patches.setattr(
            type(e.x),
            "__deepcopy__",
            Mock(side_effect=AssertionError("device array copied")),
        )
        patches.setattr(
            jax, "device_get", Mock(side_effect=AssertionError("device array fetched"))
        )
        with Debugger(trace, e.config, ()) as (layer, inputs):
            assert inputs.x is e.x
            assert layer.cached is e.x
            assert (
                layer.scope.variables()["params"]["Dense_0"]["kernel"]
                is e.variables["params"]["child"]["Dense_0"]["kernel"]
            )


@pytest.mark.parametrize("fail", [False, True])
def test_multiple_debuggers_keep_scopes_separate_and_close(experiment, fail):
    e = experiment
    stack = list(linen_module._context.module_stack)
    with pytest.raises(ValueError) if fail else nullcontext():
        with Debugger.multiple(e.trace, e.config, ["child", ""]) as (layers, inputs):
            assert len(layers) == len(inputs) == 2
            assert e.trace.call_count == 2
            assert inputs[0].x is inputs[1].x is e.x
            for index in [0, 1]:
                np.testing.assert_allclose(
                    layers[index](**inputs[index]), e.model.apply(e.variables, e.x)
                )
            with pytest.raises(RuntimeError, match="active debugger"):
                Debugger(e.trace, e.config, "child")
            if fail:
                raise ValueError("analysis failed")
    assert linen_module._context.module_stack == stack
    for layer in layers:
        with pytest.raises(RuntimeError, match="closed"):
            layer(e.x)
    with Debugger(e.trace, e.config, "child") as (layer, inputs):
        assert inputs.x is e.x


def test_multiple_capture_failure_releases_context(experiment):
    e = experiment
    with pytest.raises(LookupError):
        with Debugger.multiple(e.trace, e.config, ["child", "missing"]):
            pytest.fail("must fail during capture")
    with Debugger(e.trace, e.config, "child") as (layer, inputs):
        np.testing.assert_allclose(layer(**inputs), e.model.apply(e.variables, e.x))
