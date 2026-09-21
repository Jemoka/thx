"""Replay checks use real files and cross plan, buffer, and checkpoint boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from theseus.base import Node
from theseus.training.flywheel import pmd, stream
from theseus.training.flywheel.padded import PaddedDataset
from theseus.training.flywheel.contrastive import ContrastivePaddedDataset
from theseus.training.flywheel.strategy import Sampling, Strategy
from theseus.data.datasets import DatasetComponent
from theseus.config import configuration
from omegaconf import OmegaConf


class A(DatasetComponent):
    DATASET_KEY = "a"


class B(DatasetComponent):
    DATASET_KEY = "b"


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "PLAN_SAMPLES", 32)
    monkeypatch.setattr(pmd, "BYTES_PER_BLOCK", 128)
    monkeypatch.setattr(pmd, "BUFFER_BLOCKS", 2)
    for name, offset in [("a", 1), ("b", 10001)]:
        path = tmp_path / name
        path.mkdir()
        data = np.arange(31 * 9, dtype=np.uint32).reshape(31, 9) + offset
        data.tofile(path / "train.bin")
        np.ones_like(data, dtype=np.bool_).tofile(path / "train.bin.mask")
        (path / "shape.json").write_text(json.dumps({"train": list(data.shape)}))
        for side in ["pos", "neg"]:
            data.tofile(path / f"train.{side}.bin")
            np.ones_like(data, dtype=np.bool_).tofile(path / f"train.{side}.bin.mask")
    with configuration(OmegaConf.create({"data": {"suffix": ""}})):
        yield SimpleNamespace(hardware=SimpleNamespace(hosts=[
            SimpleNamespace(cluster=SimpleNamespace(data_dir=tmp_path))
        ]))


@pytest.mark.parametrize("style", ["pmd", "padded"])
def test_repeat_tick_resume_and_mixture(sources, style):
    def make(node, seed=17):
        with configuration(OmegaConf.create({"data": {"suffix": ""}})):
            strategy = Strategy(sources, 8, [Sampling(A, .7, style), Sampling(B, .3, style)])
        return strategy.get_async_batches(7, node=node, seed=seed)

    node = Node(name="run")
    loader = make(node)
    expected = []
    try:
        for step in range(20):
            batch = loader.get_batch()
            assert loader.get_batch() is batch
            assert loader.node is node
            assert batch["x"].shape == (7, 8)
            np.testing.assert_array_equal(batch["y"], batch["x"] + 1)
            expected.append({key: value.copy() for key, value in batch.items()})
            node.update(node.next())
        # Restoring a completed checkpoint resumes at its successor.
        for checkpoint in [0, 3, 4, 9, 18]:
            restored = make(Node(name="run", seq=checkpoint).next())
            try:
                for key, value in expected[checkpoint + 1].items():
                    np.testing.assert_array_equal(restored.get_batch()[key], value)
            finally:
                restored.close()
        assert any((batch["x"] >= 10001).any() for batch in expected)
        assert any((batch["x"] < 10001).any() for batch in expected)
    finally:
        loader.close()
    assert not loader.thread.is_alive()


@pytest.mark.parametrize("style", ["pmd", "padded"])
def test_seed_validation_and_large_batch(sources, style):
    strategy = Strategy(sources, 8, [Sampling(A, 1, style)])
    train_node, val_node = Node(name="train"), Node(name="val")
    train = strategy.get_async_batches(71, node=train_node, seed=5)
    val = strategy.get_async_batches(71, split="val", node=val_node, seed=5)
    different = strategy.get_async_batches(71, node=Node(name="other"), seed=6)
    try:
        before = val.get_batch()
        first = train.get_batch()
        assert first["x"].shape == (71, 8)  # larger than dataset and plan
        assert not np.array_equal(first["x"], different.get_batch()["x"])
        for _ in range(4):
            train_node.update(train_node.next())
            train.get_batch()
            assert val.get_batch() is before
    finally:
        train.close()
        val.close()
        different.close()


def test_pmd_tail_and_minimal_dataset(sources):
    path = sources.hardware.hosts[0].cluster.data_dir / "a"
    np.arange(9, dtype=np.uint32).tofile(path / "train.bin")
    reader = pmd.MemmapDataset(sources, 8, "a")
    batch = reader.get_batch(5)
    np.testing.assert_array_equal(batch["x"], np.tile(np.arange(8), (5, 1)))
    np.testing.assert_array_equal(batch["y"], batch["x"] + 1)


def test_padded_masks_and_contrastive_replay(sources):
    path = sources.hardware.hosts[0].cluster.data_dir / "a"
    mask = np.memmap(path / "train.bin.mask", dtype=np.bool_, mode="r+", shape=(31, 9))
    mask[:, :3] = False
    mask.flush()
    batch = PaddedDataset(sources, 8, "a").get_batch(40)
    assert (batch["y"][:, :2] == -1).all()
    assert not batch["padding_mask"][:, :3].any()
    (path / "shape.json").write_text(json.dumps({"train": {"pos": [31, 9], "neg": [31, 9]}}))
    readers = [ContrastivePaddedDataset(sources, 8, "a") for _ in range(2)]
    for _ in range(5):
        left, right = [reader.get_batch(13) for reader in readers]
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("style", ["pmd", "padded"])
def test_host_slices_match_global_stream(sources, monkeypatch, style):
    strategy = Strategy(sources, 8, [Sampling(A, 1, style)])
    monkeypatch.setattr(stream.jax, "process_count", lambda: 1)
    monkeypatch.setattr(stream.jax, "process_index", lambda: 0)
    global_batches = list(next_batch for _, next_batch in zip(range(5), stream.batches(strategy.datasets, [1], 14)))
    monkeypatch.setattr(stream.jax, "process_count", lambda: 2)
    for rank in range(2):
        monkeypatch.setattr(stream.jax, "process_index", lambda: rank)
        local = stream.batches(strategy.datasets, [1], 7)
        for global_batch in global_batches:
            for key, value in next(local).items():
                np.testing.assert_array_equal(value, global_batch[key][rank * 7:(rank + 1) * 7])


def test_prefetch_failure_and_close(sources):
    strategy = Strategy(sources, 8, [Sampling(A, 1, "padded")])
    loader = strategy.get_async_batches(0)
    try:
        with pytest.raises(ValueError, match="positive"):
            loader.get_batch()
        with pytest.raises(ValueError, match="positive"):
            loader.get_batch()
    finally:
        loader.close()
    idle = strategy.get_async_batches(2)
    idle.close()
    with pytest.raises(RuntimeError, match="closed"):
        idle.get_batch()


def test_zero_rows_are_replaced_deterministically(sources):
    path = sources.hardware.hosts[0].cluster.data_dir / "a"
    data = np.memmap(path / "train.bin", dtype=np.uint32, mode="r+", shape=(31, 9))
    data[:15] = 0
    data.flush()
    strategy = Strategy(sources, 8, [Sampling(A, 1, "padded")])
    first = strategy.get_async_batches(100, node=Node(name="a"))
    second = strategy.get_async_batches(100, node=Node(name="b"))
    try:
        batch = first.get_batch()
        assert batch["x"].any(axis=1).all()
        np.testing.assert_array_equal(batch["x"], second.get_batch()["x"])
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("style", ["pmd", "padded"])
def test_trainer_checkpoint_restores_next_batch(tmp_path, style):
    from unittest.mock import Mock

    from theseus.base import ExecutionSpec
    from theseus.data.datasets import DatasetComponent
    from theseus.model.models.base import GPT
    from theseus.quick import quick
    from theseus.training.base import BaseTrainer, BaseTrainerConfig
    from theseus.training.flywheel.strategy import Sampling

    class LocalData(DatasetComponent):
        DATASET_KEY = "replay"

    class LocalTrainer(BaseTrainer):
        MODEL = GPT
        CONFIG = BaseTrainerConfig
        DATASET = [Sampling(LocalData, 1, style)]

    spec = ExecutionSpec.local(str(tmp_path), name="replay")
    path = spec.hardware.hosts[0].cluster.data_dir / "replay"
    path.mkdir(parents=True)
    tokens = (np.arange(128 * 9, dtype=np.uint32).reshape(128, 9) % 30) + 1
    tokens.tofile(path / "train.bin")
    np.ones_like(tokens, dtype=np.bool_).tofile(path / "train.bin.mask")
    (path / "shape.json").write_text(json.dumps({"train": list(tokens.shape)}))
    with quick(tmp_path) as q:
        q.build(LocalTrainer, "replay")
        cfg = q.config
        cfg.architecture.n_layers = 1
        cfg.architecture.n_embd = 16
        cfg.architecture.n_head = 2
        cfg.architecture.intermediate_size = 32
        cfg.architecture.block_size = 8
        cfg.architecture.vocab_size = 32
        cfg.architecture.dtype.param = "float32"
        cfg.architecture.dtype.activation = "float32"
        cfg.training.batch_size = 1
        cfg.training.per_device_batch_size = 1
        cfg.training.tokens = 32
        cfg.training.validation = False
        cfg.training.evaluate = False
        cfg.training.analyze = False
        cfg.logging.remote = False
        trainer = q.create()
        assert trainer.train_dl.node is trainer.node
        assert trainer.batch() is trainer.batch()
        original = trainer.node
        # Execute a real optimizer step, then checkpoint that completed batch.
        saved_batch = trainer.batch()
        batch = trainer._to_global(trainer._reshape_batch(saved_batch))
        trainer.state, *_ = trainer._make_train_step()(
            trainer.state, batch, trainer.dropout_key, trainer.accumulate_steps
        )
        trainer.checkpoint()
        saved = trainer.node.model_copy()
        trainer.chkpt_manager.close()
        trainer.store.close()
        trainer.tick()
        assert trainer.node is original
        expected = trainer.batch()
        restored = LocalTrainer(spec, base=saved)
        borrowed_node = restored.train_dl.node
        try:
            restored.setup(resume=True)
            assert restored.node is borrowed_node
            assert restored.node.seq == saved.seq
            assert int(restored.state.step) == int(trainer.state.step)
            for key, value in saved_batch.items():
                np.testing.assert_array_equal(restored.batch()[key], value)
            train_step = Mock(wraps=restored._make_train_step())
            restored._make_train_step = Mock(return_value=train_step)
            restored.total_steps = int(restored.state.step) + 1
            restored.train()
            train_step.assert_called_once()
            for key, value in restored._reshape_batch(expected).items():
                np.testing.assert_array_equal(train_step.call_args.args[1][key], value)
            assert restored.node.seq == saved.seq + 2
        finally:
            restored.train_dl.close()
            restored.val_dl.close()
            restored.finish()
            trainer.train_dl.close()
            trainer.val_dl.close()


@pytest.mark.parametrize("style", ["padded", "pmd"])
@pytest.mark.parametrize("empty_rows", [False, True])
def test_async_host_partitions_replay_after_topology_change(sources, monkeypatch, style, empty_rows):
    if empty_rows:
        for name in ("a", "b"):
            path = sources.hardware.hosts[0].cluster.data_dir / name / "train.bin"
            data = np.memmap(path, dtype=np.uint32, mode="r+", shape=(31, 9))
            data[:8] = 0
            data.flush()
    sources.hardware.hosts *= 4
    whole = []
    # Same global batch, with different numbers of hosts and unrelated nonces.
    for hosts in (1, 2, 4):
        monkeypatch.setattr(stream.jax, "process_count", lambda: hosts)
        pieces = []
        for rank in range(hosts):
            monkeypatch.setattr(stream.jax, "process_index", lambda: rank)
            strategy = Strategy(sources, 8, [Sampling(A, .7, style), Sampling(B, .3, style)])
            node = Node(name="run", nonce=f"rank{rank}", seq=3)
            loader = strategy.get_async_batches(28 // hosts, node=node, seed=17)
            try:
                rank_batches = []
                for _ in range(6):
                    batch = loader.get_batch()
                    assert loader.get_batch() is batch
                    rank_batches.append({key: value.copy() for key, value in batch.items()})
                    node.update(node.next())
                pieces.append(rank_batches)
            finally:
                loader.close()
        combined = [{key: np.concatenate([part[step][key] for part in pieces]) for key in pieces[0][step]} for step in range(6)]
        if hosts == 1:
            whole = combined
        else:
            for expected, actual in zip(whole, combined):
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
