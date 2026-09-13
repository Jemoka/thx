"""Small multi-device correctness tests; run with JAX_NUM_CPU_DEVICES=4."""

from types import SimpleNamespace
from functools import partial

import flax.linen as nn
from flax.core import meta
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import optax
import pytest

from theseus.base import Axis, ShardingPlan, ShardingPolicy, SUPPORTED_CHIPS, Topology
from theseus.checkpoint import CheckpointManager
from theseus.inference.base import InferenceJob
from theseus.model.module import Module
from theseus.model import models
from theseus.model.axes import Axes
from theseus.config import build, configuration, configure
from theseus.training.base import BaseTrainer
from theseus.training.optimizers import AdamW, Muon
from theseus.training.optimizers.adamw import AdamWConfig
from theseus.training.optimizers.muon import MuonConfig

pytestmark = pytest.mark.skipif(
    jax.device_count() != 4, reason="Run with JAX_NUM_CPU_DEVICES=4"
)


class TinyModel(Module):
    rows: int = 8
    columns: int = 8
    annotated: bool = True

    @property
    def sharding(self):
        return ShardingPlan(tp=(("in", None), ("out", Axis.SHARD)))

    @classmethod
    def components(cls):
        return []

    @nn.compact
    def __call__(self, x, y=None, padding_mask=None, deterministic=False):
        init = nn.initializers.normal(0.05)
        if self.annotated:
            init = nn.with_logical_partitioning(init, ("in", "out"))
        kernel = self.param("kernel", init, (self.rows, self.columns))
        bias = self.param("bias", nn.initializers.zeros, (self.columns,))
        logits = x.astype(jnp.float32) @ kernel + bias
        loss = jnp.mean(jnp.square(logits - y)) if y is not None else jnp.array(0.0)
        return logits, loss


@pytest.fixture
def trainer_factory():
    def create(
        policy, optimizer="adamw", rows=8, columns=8, annotated=True, model=None
    ):
        trainer = object.__new__(BaseTrainer)
        trainer.model = (
            model
            if model is not None
            else TinyModel(rows=rows, columns=columns, annotated=annotated)
        )
        trainer.args = SimpleNamespace(block_size=rows, activation_dtype="float32")
        trainer.mesh = Topology.new(SUPPORTED_CHIPS["cpu"], shard=policy).mesh
        trainer.spec = SimpleNamespace(topology=SimpleNamespace(shard=policy))
        trainer.init_key = jax.random.PRNGKey(7)
        trainer.optimizer = (
            AdamW.optimizer(0.001, AdamWConfig())
            if optimizer == "adamw"
            else Muon.optimizer(0.001, MuonConfig())
            if optimizer == "muon"
            else optax.sgd(0.001)
        )
        trainer._init_counters_and_eval = lambda: None
        trainer.initialize()
        return trainer

    return create


def assert_values(actual, expected, tolerance=2e-5):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("optimizer", ["adamw", "muon", "sgd"])
@pytest.mark.parametrize(
    "tp,zero,fsdp",
    [
        (1, False, False),
        (1, True, False),
        (2, False, False),
        (2, True, False),
        (1, True, True),
        (2, True, True),
        (4, True, False),
    ],
)
def test_updates_match_replicated_ddp(trainer_factory, optimizer, tp, zero, fsdp):
    baseline = trainer_factory(ShardingPolicy(zero=False), optimizer)
    trainer = trainer_factory(ShardingPolicy(tp=tp, zero=zero, fsdp=fsdp), optimizer)
    host = {
        "x": np.arange(64, dtype=np.float32).reshape(2, 4, 8) / 64,
        "y": np.ones((2, 4, 8), dtype=np.float32) * 0.2,
        "padding_mask": np.ones((2, 4, 8), dtype=np.int32),
    }
    runs = [
        (
            t,
            t._make_train_step(),
            jax.device_put(host, NamedSharding(t.mesh, P(None, Axis.BATCH, None))),
        )
        for t in (baseline, trainer)
    ]
    for current, step, batch in runs:
        # Forward is rematerialized independently of the model architecture.
        assert "remat2" in str(
            jax.make_jaxpr(step)(current.state, batch, jax.random.PRNGKey(9), 2)
        )
    for _ in range(3):
        outputs = []
        for t, step, batch in runs:
            t.state, loss, metadata, norm = step(
                t.state, batch, jax.random.PRNGKey(9), 2
            )
            outputs.append((loss, norm))
        assert_values(outputs[0], outputs[1])
        assert_values(trainer.state.params, baseline.state.params)
        assert_values(trainer.state.opt_state, baseline.state.opt_state)
    expected = P(None, (Axis.SHARD, Axis.BATCH)) if fsdp else P(None, Axis.SHARD)
    assert trainer.state.params["kernel"].value.sharding.spec == expected
    assert trainer.state.params["bias"].sharding.is_fully_replicated
    for path, leaf in jax.tree_util.tree_flatten_with_path(trainer.state.opt_state)[0]:
        if "kernel" in jax.tree_util.keystr(path) and leaf.shape == (8, 8):
            assert leaf.sharding.spec == (
                P(None, (Axis.SHARD, Axis.BATCH)) if zero else P(None, Axis.SHARD)
            )


@pytest.mark.parametrize("fsdp", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("accumulation", [1, 4])
def test_checkpointing_preserves_updates(trainer_factory, fsdp, dtype, accumulation):
    policy = ShardingPolicy(tp=2, fsdp=fsdp)
    baseline = trainer_factory(policy)
    bounded = trainer_factory(policy.model_copy(update={"activation_checkpointing": True}))
    host = {
        "x": np.arange(accumulation * 32, dtype=np.float32).reshape(accumulation, 4, 8) / 64,
        "y": np.ones((accumulation, 4, 8), dtype=np.float32) * 0.2,
        "padding_mask": np.ones((accumulation, 4, 8), dtype=np.int32),
    }
    runs = []
    for trainer in (baseline, bounded):
        trainer.args.activation_dtype = dtype
        batch = jax.device_put(host, NamedSharding(trainer.mesh, P(None, Axis.BATCH, None)))
        runs.append((trainer, trainer._make_train_step(), batch))
    for trainer, step, batch in runs:
        graph = str(jax.make_jaxpr(step)(
            trainer.state, batch, jax.random.PRNGKey(9), accumulation
        ))
        assert ("optimization_barrier" in graph) == fsdp
        assert ("policy=None" in graph) == trainer.spec.topology.shard.activation_checkpointing
    for _ in range(3):
        outputs = []
        for trainer, step, batch in runs:
            trainer.state, *measurements = step(
                trainer.state, batch, jax.random.PRNGKey(9), accumulation
            )
            outputs.append(measurements)
        assert_values(outputs[0], outputs[1])
        assert_values(bounded.state.params, baseline.state.params)
        assert_values(bounded.state.opt_state, baseline.state.opt_state)
        for actual, expected in zip(jax.tree.leaves(bounded.state), jax.tree.leaves(baseline.state), strict=True):
            assert actual.sharding == expected.sharding


@pytest.mark.parametrize("fsdp", [False, True])
def test_unannotated_parameters_remain_replicated(trainer_factory, fsdp):
    trainer = trainer_factory(ShardingPolicy(tp=2, fsdp=fsdp), annotated=False)
    assert all(x.sharding.is_fully_replicated for x in jax.tree.leaves(trainer.state))


@pytest.mark.parametrize(
    "policy",
    [
        ShardingPolicy(tp=4, zero=False),
        ShardingPolicy(),
        ShardingPolicy(tp=2, fsdp=True),
    ],
)
def test_nondivisible_parameter_partition_fails_before_allocation(
    trainer_factory, policy
):
    with pytest.raises(ValueError, match="kernel.*partitioned 4 times"):
        trainer_factory(policy, columns=6)


@pytest.mark.parametrize("rows,columns", [(8, 4), (4, 8)])
def test_muon_factored_moments_drop_reduced_axis(trainer_factory, rows, columns):
    trainer = trainer_factory(ShardingPolicy(tp=2), "muon", rows=rows, columns=columns)
    batch = jax.device_put(
        {
            "x": np.ones((1, 4, rows), np.float32),
            "y": np.zeros((1, 4, columns), np.float32),
            "padding_mask": np.ones((1, 4, columns), np.int32),
        },
        NamedSharding(trainer.mesh, P(None, Axis.BATCH, None)),
    )
    step = trainer._make_train_step()
    hlo = step.lower(trainer.state, batch, jax.random.PRNGKey(0), 1).compile().as_text()
    assert "all-gather" in hlo.lower()
    trainer.state, *_ = step(trainer.state, batch, jax.random.PRNGKey(0), 1)
    for path, leaf in jax.tree_util.tree_flatten_with_path(trainer.state.opt_state)[0]:
        if "second_moment" in jax.tree_util.keystr(
            path
        ) and "kernel" in jax.tree_util.keystr(path):
            assert leaf.shape == ((rows, 1) if rows >= columns else (1, columns))
            assert leaf.sharding.spec == (
                P(None, None) if rows >= columns else P(None, (Axis.SHARD, Axis.BATCH))
            )


class TokenModel(TinyModel):
    @nn.compact
    def __call__(self, x, y=None, **kwargs):
        kernel = self.param(
            "kernel",
            nn.with_logical_partitioning(nn.initializers.normal(0.05), ("in", "out")),
            (8, 8),
        )
        return jax.nn.one_hot(x, 8) @ kernel, jnp.array(0.0)


@pytest.mark.parametrize("tp", [1, 2])
def test_fsdp_gathers_weights_for_inference(trainer_factory, tp):
    policy = ShardingPolicy(tp=tp, fsdp=True)
    inference = object.__new__(InferenceJob)
    inference.model = TokenModel()
    inference.mesh = Topology.new(SUPPORTED_CHIPS["cpu"], shard=policy).mesh
    inference.spec = SimpleNamespace(topology=SimpleNamespace(shard=policy))
    inference.tx = optax.identity()
    inference._trainer = None
    inference.key = jax.random.PRNGKey(7)
    inference.block_size = 8
    inference.initialize()
    x = jax.device_put(
        np.ones((4, 2), np.int32), NamedSharding(inference.mesh, P(Axis.BATCH, None))
    )
    masks = jax.device_put(np.ones((4, 2), bool), x.sharding)
    generate = jax.jit(
        lambda state, x, mask: inference._autoregress(
            state, jax.random.PRNGKey(0), x, mask, 4, 0.0, 1.0
        ),
        in_shardings=(inference.state_sharding, x.sharding, x.sharding),
    )
    compiled = generate.lower(inference.state, x, masks).compile()
    assert "all-gather" in compiled.as_text().lower()
    assert generate(inference.state, x, masks).shape == (4, 4)


@pytest.mark.parametrize(
    "source_policy,target_policy",
    [
        (ShardingPolicy(zero=False), ShardingPolicy(tp=2, fsdp=True)),
        (ShardingPolicy(tp=2, fsdp=True), ShardingPolicy()),
    ],
)
def test_restore_changes_layout_and_upgrades_metadata(
    tmp_path, trainer_factory, source_policy, target_policy
):
    source = trainer_factory(source_policy)
    target = trainer_factory(target_policy)
    # Simulate checkpoints written with the old physical metadata wrapper.
    old_state = jax.tree.map(
        lambda leaf: (
            nn.Partitioned(leaf.value, leaf.names)
            if isinstance(leaf, nn.LogicallyPartitioned)
            else leaf
        ),
        source.state,
        is_leaf=meta.is_axis_metadata,
    )
    manager = CheckpointManager()
    try:
        manager.save(old_state, tmp_path / "checkpoint")
        manager.checkpointer.wait_until_finished()
        restored, missing = manager.restore(tmp_path / "checkpoint", target.template)
        assert missing is None
        assert_values(restored.params, source.state.params)
        assert_values(restored.opt_state, source.state.opt_state)
        assert isinstance(restored.params["kernel"], nn.LogicallyPartitioned)
        assert (
            restored.params["kernel"].value.sharding
            == target.state.params["kernel"].value.sharding
        )
    finally:
        manager.close()


@pytest.mark.parametrize(
    "policy",
    [
        ShardingPolicy(tp=2, zero=False),
        ShardingPolicy(),
        ShardingPolicy(fsdp=True),
        ShardingPolicy(tp=2, fsdp=True),
    ],
)
def test_validation_uses_trainer_sharding_context(trainer_factory, policy):
    baseline = trainer_factory(ShardingPolicy(zero=False))
    trainer = trainer_factory(policy)
    host = {
        "x": np.arange(64, dtype=np.float32).reshape(2, 4, 8) / 64,
        "y": np.ones((2, 4, 8), np.float32) * 0.2,
        "padding_mask": np.ones((2, 4, 8), np.int32),
    }
    results = []
    for current in (baseline, trainer):
        data_sharding = NamedSharding(current.mesh, P(None, Axis.BATCH, None))
        batch = jax.device_put(host, data_sharding)
        validate = jax.jit(
            partial(current.val_step, sharding=current.sharding_context),
            in_shardings=(current.state_sharding, data_sharding),
        )
        compiled = validate.lower(current.state, batch).compile()
        if current is trainer and policy.fsdp:
            assert "all-gather" in compiled.as_text().lower()
        results.append(compiled(current.state, batch))
    assert_values(results[1], results[0])


def test_compiled_steps_do_not_retain_trainer_and_donate_state(trainer_factory):
    import gc
    import weakref

    trainer = trainer_factory(ShardingPolicy(tp=2, fsdp=True))
    state = trainer.state
    context = trainer.sharding_context
    data_sharding = NamedSharding(trainer.mesh, P(None, Axis.BATCH, None))
    batch = jax.device_put(
        {
            "x": np.ones((1, 4, 8), np.float32),
            "y": np.zeros((1, 4, 8), np.float32),
            "padding_mask": np.ones((1, 4, 8), np.int32),
        },
        data_sharding,
    )
    key = jax.random.PRNGKey(9)
    assert trainer.train_step.__self__ is type(trainer)
    assert trainer.val_step.__self__ is type(trainer)
    train = trainer._make_train_step()
    validate = jax.jit(
        partial(trainer.val_step, sharding=context),
        in_shardings=(trainer.state_sharding, data_sharding),
    )
    train.lower(state, batch, key, 1).compile()
    expected = validate(state, batch)
    reference = weakref.ref(trainer)
    del trainer
    gc.collect()
    assert reference() is None
    assert_values(validate(state, batch), expected)
    parameter = state.params["kernel"].value
    updated, *_ = train(state, batch, key, 1)
    jax.block_until_ready(updated)
    assert parameter.is_deleted()
    assert not batch["x"].is_deleted()
    assert int(updated.step) == 1


def test_new_context_retraces_without_mutating_existing_step(trainer_factory):
    from dataclasses import FrozenInstanceError, replace

    trainer = trainer_factory(ShardingPolicy(tp=2))
    context = trainer.sharding_context
    with pytest.raises(FrozenInstanceError):
        context.parameter_fwdbwd_mapping = ()
    assert isinstance(context.parameter_storage_sharding, tuple)
    assert isinstance(context.parameter_update_sharding, tuple)
    assert all(
        isinstance(leaf, NamedSharding) for leaf in context.parameter_storage_sharding
    )
    assert hash(context)
    batch = jax.device_put(
        {
            "x": np.ones((1, 4, 8), np.float32),
            "y": np.zeros((1, 4, 8), np.float32),
            "padding_mask": np.ones((1, 4, 8), np.int32),
        },
        NamedSharding(trainer.mesh, P(None, Axis.BATCH, None)),
    )
    key = jax.random.PRNGKey(9)
    original = trainer._make_train_step()
    original_graph = str(jax.make_jaxpr(original)(trainer.state, batch, key, 1))
    assert "remat2" in original_graph
    trainer.sharding_context = replace(
        context,
        parameter_fwdbwd_mapping=tuple(
            (axis, None) for axis, _ in context.parameter_fwdbwd_mapping
        ),
    )
    replacement = trainer._make_train_step()
    replacement_graph = str(jax.make_jaxpr(replacement)(trainer.state, batch, key, 1))
    assert "remat2" in replacement_graph
    assert replacement_graph != original_graph
    assert str(jax.make_jaxpr(original)(trainer.state, batch, key, 1)) == original_graph


@pytest.mark.parametrize(
    "model_class",
    [
        getattr(models, name)
        for name in models.__all__
        if issubclass(getattr(models, name), models.GPT)
    ],
    ids=lambda cls: cls.__name__,
)
def test_gpt_subclasses_preserve_embedding_zero_rules(model_class):
    plan = model_class().sharding
    assert dict(plan._tp)[Axes.VOCAB.value] is None
    assert dict(plan._tp_x_zero)[Axes.VOCAB.value] == Axis.BATCH
    assert dict(plan._tp_x_zero)[Axes.BLOCK_SIZE.value] == Axis.BATCH
    # Every TP dimension must also keep its optimizer-state partition.
    for axis, physical in plan.tp:
        if physical == Axis.SHARD:
            assert dict(plan._tp_x_zero)[axis] == (Axis.SHARD, Axis.BATCH)


@pytest.mark.parametrize(
    "policy",
    [
        ShardingPolicy(tp=2, zero=False),
        ShardingPolicy(),
        ShardingPolicy(fsdp=True),
        ShardingPolicy(tp=2, fsdp=True),
    ],
)
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_gpt_embedding_storage_and_updates(trainer_factory, policy, activation_checkpointing):
    policy = policy.model_copy(update={"activation_checkpointing": activation_checkpointing})
    cfg = build(*models.GPT.gather())
    cfg.architecture.n_layers = 1
    cfg.architecture.n_embd = 16
    cfg.architecture.n_head = 4
    cfg.architecture.vocab_size = 32
    cfg.architecture.block_size = 8
    cfg.architecture.rope = False
    cfg.architecture.dropout = 0.0
    cfg.architecture.dtype.activation = "float32"
    cfg.architecture.dtype.param = "float32"
    tokens = np.arange(32, dtype=np.int32).reshape(1, 4, 8)
    host = {"x": tokens, "y": (tokens + 1) % 32, "padding_mask": np.ones_like(tokens)}
    with configuration(cfg):
        baseline = trainer_factory(
            ShardingPolicy(zero=False), model=configure(models.GPT)
        )
        trainer = trainer_factory(policy, model=configure(models.GPT))
        for current in (baseline, trainer):
            step = current._make_train_step()
            batch = jax.device_put(
                host, NamedSharding(current.mesh, P(None, Axis.BATCH, None))
            )
            for _ in range(2):
                current.state, *_ = step(current.state, batch, jax.random.PRNGKey(4), 1)
        assert_values(trainer.state.params, baseline.state.params)
        assert_values(trainer.state.opt_state, baseline.state.opt_state)
        for name in ("wte", "wpe"):
            parameter = trainer.state.params[name].value
            assert parameter.sharding.spec == P(
                Axis.BATCH if policy.fsdp else None, None
            )
            moments = [
                leaf
                for path, leaf in jax.tree_util.tree_flatten_with_path(
                    trainer.state.opt_state
                )[0]
                if name in jax.tree_util.keystr(path)
            ]
            assert moments
            for moment in moments:
                assert moment.sharding.spec == P(
                    Axis.BATCH if policy.zero else None, None
                )
