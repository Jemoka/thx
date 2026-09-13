from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import theseus.registry as registry
from theseus.base import ShardingPlan, ExecutionSpec, PyTree
from theseus.config import build, configuration, field
from theseus.evaluation.base import Evaluation
from theseus.inference.base import InferenceJob
from theseus.job import RestoreableJob
from theseus.model.module import Module
from theseus.store import ObjectStore, RecordStore
from theseus.training.base import BaseTrainer, BaseTrainerConfig


class TinyModel(Module):
    """Small token model used to exercise the complete x job stack."""

    extra_mlp: bool = field("architecture/extra_mlp", default=False)

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    @classmethod
    def components(cls) -> list[type[Any]]:
        return []

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        y: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        del deterministic
        hidden = nn.Embed(
            num_embeddings=4,
            features=4,
            dtype=self._activation_dtype,
            param_dtype=self._param_dtype,
        )(x)
        logits = nn.Dense(
            features=4,
            dtype=self._activation_dtype,
            param_dtype=self._param_dtype,
        )(hidden)
        if self.extra_mlp:
            logits = logits + nn.Dense(
                features=4,
                dtype=self._activation_dtype,
                param_dtype=self._param_dtype,
                name="new_mlp",
            )(hidden)
        if y is None:
            return logits, jnp.zeros((), dtype=logits.dtype)

        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        mask = (
            jnp.ones_like(losses)
            if padding_mask is None
            else padding_mask.astype(losses.dtype)
        )
        return logits, (losses * mask).sum() / mask.sum()


@dataclass
class TinyTrainerConfig(BaseTrainerConfig):
    batch_size: int = field("training/batch_size", default=1)
    per_device_batch_size: int = field("training/per_device_batch_size", default=1)
    total_tokens: int = field("training/tokens", default=8)
    validate: bool = field("training/validation", default=False)
    evaluate: bool = field("training/evaluate", default=True)
    block_size: int = field("architecture/block_size", default=4)
    report_interval: int = field("logging/report_interval", default=1)
    checkpoint_interval: int = field("logging/checkpoint_interval", default=2)
    validation_interval: int = field("logging/validation_interval", default=1)


class TinyTrainer(BaseTrainer[TinyTrainerConfig, TinyModel]):
    MODEL = TinyModel
    CONFIG = TinyTrainerConfig
    DATASET = []

    def _init_data(self, spec: ExecutionSpec) -> None:
        del spec

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        del slice
        return {
            "x": np.asarray([[0, 1, 2, 3]], dtype=np.int32),
            "y": np.asarray([[1, 2, 3, 0]], dtype=np.int32),
            "padding_mask": np.ones((1, 4), dtype=np.bool_),
        }


class RestoreOnlyTinyTrainer(TinyTrainer):
    def run(self) -> None:
        pass


class TinyEvaluation(Evaluation):
    @property
    def name(self) -> str:
        return "tiny/evaluation"

    def __len__(self) -> int:
        return 1

    def __call__(
        self,
        inference: InferenceJob[Any, TinyModel],
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> float:
        del encoding, reduce, return_intermediates, kwargs
        step = int(jax.device_get(inference.state.step))
        inference.log({"tiny/evaluated_step": step})
        return float(step)


TinyTrainer.EVALUATION = [TinyEvaluation]


def test_trainer_uses_static_data_and_evaluation_components() -> None:
    cfg = build(*TinyTrainer.config())

    assert "dataset" not in cfg.training
    assert "evaluations" not in cfg.eval
    assert TinyTrainer.EVALUATION == [TinyEvaluation]


class TinyInference(InferenceJob[TinyTrainerConfig, TinyModel]):
    MODEL = TinyModel

    @classmethod
    def config(cls) -> list[type[Any]]:
        return [TinyTrainerConfig, *TinyModel.gather()]

    def run(self) -> None:
        self.log({"tiny/inference_step": int(jax.device_get(self.state.step))})


def test_x_stack_trains_evaluates_checkpoints_and_restores(
    tmp_path, monkeypatch
) -> None:
    job_key = "tests/x/tiny"
    inference_key = "tests/x/tiny-inference"
    evaluation_key = "tests/x/tiny-evaluation"
    monkeypatch.setattr(registry, "ensure_registered", lambda: None)
    monkeypatch.setitem(registry.JOBS, job_key, None)
    monkeypatch.setitem(registry.JOBS, inference_key, None)
    monkeypatch.setitem(registry.EVALUATIONS, evaluation_key, None)
    registry.job(job_key)(TinyTrainer)
    registry.job(inference_key)(TinyInference)
    registry.evaluation(evaluation_key)(TinyEvaluation)
    monkeypatch.setattr("theseus.evaluation.base.get_tokenizer", object)

    cfg = build(*TinyTrainer.config())
    cfg.architecture.dtype.activation = "float32"
    cfg.training.validation = True
    spec = ExecutionSpec.local(
        str(tmp_path),
        name="tiny",
        project="tests",
        group="x",
    ).model_copy(update={"execution_id": "tiny-execution", "tag": "train"})

    with configuration(cfg):
        trainer = TinyTrainer(spec)
        trainer()
    expected_params = jax.device_get(trainer.state.params)

    store = RecordStore(spec.hardware)
    checkpoints = store.query().name("tests.x.tiny").checkpoint().all()
    artifacts = store.query().name("tests.x.tiny").artifact().all()
    assert [node.seq for node in checkpoints] == [0, 1]
    assert artifacts == checkpoints
    final = checkpoints[-1]
    row = store.query().node(final).select(raw=True)[0]
    finished = (
        store.query().execution("tiny-execution").tag("train").finished().latest().all()
    )
    record_row = store.get_record(final)
    store.close()

    assert row["_x_job"] == job_key
    assert row["_x_checkpoint"] is True
    assert row["_x_record"] is True
    assert row["_x_execution"] == "tiny-execution"
    assert row["_x_tag"] == "train"
    assert finished == [trainer.node.model_copy(update={"parent": None})]
    assert row["tiny/evaluated_step"] == 2
    assert row["tiny/evaluation"] == 2.0
    blob = row["blob"]
    assert isinstance(blob, Path)
    record = record_row["payload"][0]
    assert record == {"intermediates": {}, "plots": {}}

    restored, restored_cfg = RestoreableJob.from_node(
        final,
        spec,
        runtime_cfg=cfg,
        resume=True,
    )
    assert isinstance(restored, TinyTrainer)
    with configuration(restored_cfg):
        restored(resume=True)

    assert restored.node == final.next()
    assert int(jax.device_get(restored.state.step)) == 2
    for actual, expected in zip(
        jax.tree.leaves(restored.state.params),
        jax.tree.leaves(expected_params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    inference_spec = spec.model_copy(update={"name": "tiny-inference"})
    with configuration(cfg):
        inference = TinyInference(inference_spec, base=final)
        inference()

    assert inference.node.parent == final.serialize()
    assert int(jax.device_get(inference.state.step)) == 0
    for actual, expected in zip(
        jax.tree.leaves(inference.state.params),
        jax.tree.leaves(expected_params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    object_store = ObjectStore(spec.hardware)
    inference_nodes = (
        object_store.query()
        .name("tests.x.tiny-inference")
        .has("tiny/inference_step")
        .all()
    )
    parents = object_store.parents(inference.node)
    object_store.close()
    assert inference_nodes == [inference.node.model_copy(update={"parent": None})]
    assert parents == [final]


def test_x_stack_restores_into_model_with_new_mlp(tmp_path, monkeypatch) -> None:
    job_key = "tests/x/surgery-source"
    monkeypatch.setattr(registry, "ensure_registered", lambda: None)
    monkeypatch.setitem(registry.JOBS, job_key, None)
    registry.job(job_key)(TinyTrainer)

    source_cfg = build(*TinyTrainer.config())
    source_cfg.architecture.dtype.activation = "float32"
    source_cfg.training.evaluate = False
    source_spec = ExecutionSpec.local(
        str(tmp_path),
        name="surgery-source",
        project="tests",
        group="x",
    )
    with configuration(source_cfg):
        source = TinyTrainer(source_spec)
        source()

    store = ObjectStore(source_spec.hardware)
    checkpoint = store.query().name("tests.x.surgery-source").checkpoint().all()[-1]
    store.close()

    target_cfg = build(*TinyTrainer.config())
    target_cfg.architecture.dtype.activation = "float32"
    target_cfg.architecture.extra_mlp = True
    target_cfg.training.evaluate = False
    target_spec = source_spec.model_copy(update={"name": "surgery-target"})
    with configuration(target_cfg):
        target = RestoreOnlyTinyTrainer(target_spec, base=checkpoint)
        target.initialize()
        initialized = jax.device_get(target.state)
        target()

    for name in ("Embed_0", "Dense_0"):
        jax.tree.map(
            np.testing.assert_array_equal,
            target.state.params[name],
            source.state.params[name],
        )
    jax.tree.map(
        np.testing.assert_array_equal,
        target.state.params["new_mlp"],
        initialized.params["new_mlp"],
    )

    source_adam = source.state.opt_state[1][0]
    target_adam = target.state.opt_state[1][0]
    initialized_adam = initialized.opt_state[1][0]
    np.testing.assert_array_equal(target_adam.count, source_adam.count)
    for name in ("Embed_0", "Dense_0"):
        jax.tree.map(
            np.testing.assert_array_equal,
            target_adam.mu[name],
            source_adam.mu[name],
        )
        jax.tree.map(
            np.testing.assert_array_equal,
            target_adam.nu[name],
            source_adam.nu[name],
        )
    jax.tree.map(
        np.testing.assert_array_equal,
        target_adam.mu["new_mlp"],
        initialized_adam.mu["new_mlp"],
    )
    jax.tree.map(
        np.testing.assert_array_equal,
        target_adam.nu["new_mlp"],
        initialized_adam.nu["new_mlp"],
    )

    updated = target.state.apply_gradients(
        grads=jax.tree.map(jnp.ones_like, target.state.params)
    )
    assert (
        int(jax.device_get(updated.step)) == int(jax.device_get(target.state.step)) + 1
    )
