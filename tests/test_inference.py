from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import jax
import pytest

from theseus.base import Node
from theseus.config import build, configuration
from theseus.evaluation.base import Evaluator
from theseus.evaluation.base import EvaluatorConfig
from theseus.data.tokenizer import TokenizerConfig
from theseus.inference.base import InferenceJob


def _trainer() -> Any:
    return SimpleNamespace(
        spec=Mock(),
        store=Mock(),
        base=None,
        key=jax.random.PRNGKey(0),
        mesh=Mock(),
        state_sharding=Mock(),
        replicas=4,
        local_replicas=4,
        per_device_batch_size=2,
        args=SimpleNamespace(block_size=16),
        model=Mock(),
        EVALUATION=[],
        node=Node(name="train", nonce="abcdef"),
        state=object(),
    )


def test_trainer_owned_inference_follows_node_and_state() -> None:
    trainer = _trainer()
    inference = InferenceJob.from_trainer(trainer)

    next_state = object()
    trainer.node.update(trainer.node.next())
    trainer.state = next_state
    inference.log({"eval/score": 0.5})

    assert inference.node == trainer.node
    assert inference.state is next_state
    trainer.store.value.assert_called_once_with(
        trainer.node,
        {"eval/score": 0.5},
    )


def test_trainer_owned_inference_cannot_advance_or_rebind() -> None:
    trainer = _trainer()
    inference = InferenceJob.from_trainer(trainer)

    with pytest.raises(RuntimeError, match="cannot tick independently"):
        inference.tick()
    with pytest.raises(RuntimeError, match="cannot replace its state"):
        inference.state = cast(Any, object())

    trainer.store.value.assert_not_called()


def test_standalone_inference_owns_its_clock() -> None:
    inference = cast(Any, object.__new__(InferenceJob))
    inference._trainer = None
    inference.node = Node(name="inference", nonce="abcdef")
    inference.store = Mock()
    inference._setup_complete = True
    written = []
    inference.store.value.side_effect = lambda node, kv: written.append((node.model_copy(), kv))

    inference.tick()

    assert written == [(Node(name="inference", nonce="abcdef"), {})]
    assert inference.node.seq == 1


def test_evaluator_borrows_trainer_identity(monkeypatch) -> None:
    trainer = _trainer()
    tokenizer = object()
    monkeypatch.setattr("theseus.evaluation.base.get_tokenizer", lambda: tokenizer)

    with configuration(build(EvaluatorConfig, TokenizerConfig)):
        evaluator = Evaluator.from_trainer(trainer)
    trainer.node.update(trainer.node.next())

    assert evaluator.node == trainer.node
    assert evaluator.state is trainer.state
    assert evaluator.store is trainer.store
    assert evaluator.encoding is tokenizer
