from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from theseus.base import ExecutionSpec, Node
from theseus.config import build, configuration, configure, field
from theseus.data.datasets import Dataset, DatasetConfig, StreamingPretrainingDataset
from theseus.data.datasets import FineWeb
from theseus.data.datasets.fineweb import FineWebConfig
from theseus.data.tokenize import TokenizeVariableDatasetJob
from theseus.evaluation.base import Evaluation, Evaluator
from theseus.store import ObjectStore
from theseus.training.base import BaseTrainer
from theseus.training.flywheel.strategy import DatasetStyle, Sampling, Strategy


@dataclass
class SourceConfig:
    source: str = field("data/raw/source", default="test")


@dataclass
class TinyEvaluationConfig:
    source: str = field("eval/tiny/source", default="test")


class RawDataset(Dataset[str]):
    DATASET_KEY = "raw"
    CONFIG = SourceConfig

    def __init__(self) -> None:
        self.source = configure(self.CONFIG).source

    def __getitem__(self, index: int) -> str:
        return str(index)

    def __len__(self) -> int:
        return 1


class TinyEvaluation(Evaluation):
    CONFIG = TinyEvaluationConfig

    @property
    def name(self) -> str:
        return "tiny"

    def __len__(self) -> int:
        return 0

    def __call__(self, *args: object, **kwargs: object) -> float:
        return 0.0


class Stream(StreamingPretrainingDataset):
    DATASET_KEY = "stream"

    def __iter__(self) -> Iterator[str]:
        yield from ("a", "bb", "ccc")


class StreamJob(TokenizeVariableDatasetJob):
    DATASET = Stream


class Tokenizer:
    eot_token = 0

    def encode_ordinary(self, value: str) -> list[int]:
        return [len(value)]


def test_component_configs_are_aggregated() -> None:
    sampling = Sampling(RawDataset, rate=1.0)

    assert Strategy.config([sampling]) == [DatasetConfig, SourceConfig]
    assert BaseTrainer._normalize_ds(sampling)[0] is sampling
    assert BaseTrainer._normalize_ds(RawDataset) == [Sampling(RawDataset, rate=1.0)]
    assert Evaluator.config([TinyEvaluation])[-1] is TinyEvaluationConfig
    assert FineWeb.DATASET_KEY == "fineweb"


@pytest.mark.parametrize(
    ("rates", "implicit_count", "expected_rate"),
    [
        ([], 2, 0.5),
        ([0.7], 2, 0.15),
        ([1.0], 1, 0.0),
        ([0.1, 0.2, 0.4, 0.2, 0.1], 1, 0.0),
    ],
)
def test_normalize_ds_distributes_remaining_weight(
    rates: list[float], implicit_count: int, expected_rate: float
) -> None:
    explicit = [Sampling(RawDataset, rate) for rate in rates]
    result = BaseTrainer._normalize_ds(explicit + [RawDataset] * implicit_count)

    assert all(actual is original for actual, original in zip(result, explicit))
    implicit = result[len(explicit) :]
    assert len(implicit) == implicit_count
    assert [sample.rate for sample in implicit] == pytest.approx(
        [expected_rate] * implicit_count
    )
    assert all(sample.rate >= 0 and sample.style is None for sample in implicit)


@pytest.mark.parametrize("implicit_count", [0, 1])
def test_normalize_ds_rejects_overallocation(implicit_count: int) -> None:
    with pytest.raises(ValueError, match="Explicit sampling rates exceed 1"):
        BaseTrainer._normalize_ds(
            [Sampling(RawDataset, 1.2)] + [RawDataset] * implicit_count
        )


@pytest.mark.parametrize(
    ("dataset_name", "reader_module", "reader_name"),
    [
        ("Dataset", "padded", "PaddedDataset"),
        ("StreamingDataset", "padded", "PaddedDataset"),
        ("PretrainingDataset", "pmd", "MemmapDataset"),
        ("StreamingPretrainingDataset", "pmd", "MemmapDataset"),
        ("ContrastiveDataset", "contrastive", "ContrastivePaddedDataset"),
        ("FineWeb", "pmd", "MemmapDataset"),
        ("FineWebEduDedup", "pmd", "MemmapDataset"),
    ],
)
def test_sampling_infers_dataset_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    reader_module: str,
    reader_name: str,
) -> None:
    from unittest.mock import Mock
    from theseus.data import datasets

    dataset = type(
        "LocalDataset", (getattr(datasets, dataset_name),), {"DATASET_KEY": "local"}
    )
    reader = Mock()
    monkeypatch.setattr(
        f"theseus.training.flywheel.{reader_module}.{reader_name}", reader
    )
    spec = ExecutionSpec.local(str(tmp_path))
    sampling = Sampling(dataset, 1.0)

    assert sampling.style is None
    with configuration(build(DatasetConfig)):
        strategy = Strategy(spec, 8, [sampling])

    reader.assert_called_once_with(spec, 8, "local", "")
    assert strategy.datasets == [reader.return_value]


@pytest.mark.parametrize("declared_style", [None, DatasetStyle.PMD])
def test_sampling_explicit_style_overrides_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared_style: DatasetStyle | None
) -> None:
    from unittest.mock import Mock
    from theseus.data.datasets import DatasetComponent

    class LocalDataset(DatasetComponent):
        DATASET_KEY = "local"
        STYLE = declared_style

    reader = Mock()
    monkeypatch.setattr("theseus.training.flywheel.padded.PaddedDataset", reader)
    spec = ExecutionSpec.local(str(tmp_path))
    with configuration(build(DatasetConfig)):
        Strategy(spec, 8, [Sampling(LocalDataset, 1.0, "padded")])

    reader.assert_called_once_with(spec, 8, "local", "")


def test_sampling_requires_style_for_unknown_dataset(tmp_path: Path) -> None:
    from theseus.data.datasets import DatasetComponent

    class UnknownDataset(DatasetComponent):
        DATASET_KEY = "unknown"

    with configuration(build(DatasetConfig)):
        with pytest.raises(
            ValueError, match="UnknownDataset has no STYLE; set Sampling"
        ):
            Strategy(
                ExecutionSpec.local(str(tmp_path)), 8, [Sampling(UnknownDataset, 1.0)]
            )


def test_evaluation_config_is_aggregated_and_hydrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from theseus.evaluation.datasets.alpaca import AlpacaEval, AlpacaEvalConfig

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "theseus.evaluation.datasets.alpaca.load_dataset",
        lambda name, *, split: calls.append((name, split)) or [],
    )
    monkeypatch.setattr("theseus.evaluation.datasets.alpaca.get_tokenizer", object)
    cfg = build(*Evaluator.config([AlpacaEval]))
    cfg.eval.alpaca.split = "validation"

    with configuration(cfg):
        evaluation = AlpacaEval()

    assert evaluation.CONFIG is AlpacaEvalConfig
    assert calls == [("tatsu-lab/alpaca", "validation")]




def test_dataset_constructor_hydrates_owned_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: dict[str, object] = {}

    def load_dataset(name: str, snapshot: str, **kwargs: object) -> object:
        loaded.update(name=name, snapshot=snapshot, **kwargs)
        return []

    monkeypatch.setattr("theseus.data.datasets.fineweb.load_dataset", load_dataset)
    cfg = build(*FineWeb.config())
    cfg.data.fineweb.snapshot = "CC-MAIN-test"

    with configuration(cfg):
        FineWeb()

    assert FineWeb.config() == [DatasetConfig, FineWebConfig]
    assert loaded == {
        "name": "HuggingFaceFW/fineweb",
        "snapshot": "CC-MAIN-test",
        "split": "train",
        "streaming": True,
    }


@pytest.mark.parametrize("suffix", ["", "old"])
def test_sampling_resolves_existing_key_suffix_layout(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / "data" / ("raw" if suffix == "" else f"raw_{suffix}")
    path.mkdir(parents=True)
    (path / "shape.json").write_text('{"train": [1, 2]}')
    np.asarray([[1, 2]], dtype=np.uint32).tofile(path / "train.bin")
    np.asarray([[True, True]], dtype=np.bool_).tofile(path / "train.bin.mask")
    cfg = build(DatasetConfig)
    cfg.data.suffix = suffix
    spec = ExecutionSpec.local(str(tmp_path))

    with configuration(cfg):
        strategy = Strategy(
            spec,
            block_size=1,
            mixture=[Sampling(RawDataset, rate=1.0, style=DatasetStyle.PADDED)],
        )

    assert strategy.datasets[0].path == path  # type: ignore[attr-defined]


@pytest.mark.parametrize("resume", [False, True])
def test_tokenization_progress_and_complete_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    monkeypatch.setattr("theseus.data.tokenize.get_tokenizer", Tokenizer)
    cfg = build(*StreamJob.config())
    spec = ExecutionSpec.local(str(tmp_path), name="stream")

    with configuration(cfg):
        job = StreamJob(spec)
        job()

    store = ObjectStore(spec.hardware)
    complete = (
        store.query().name("general.default.stream").has("data/complete").all()[-1]
    )
    state = store.query().node(complete).select()[0]
    store.close()
    assert state["data/dataset"] == "stream"
    assert state["data/samples"] == 3
    assert state["data/complete"] is True
    assert not (tmp_path / "data" / "stream" / "progress.json").exists()
    assert not (tmp_path / "data" / "stream" / "config.json").exists()

    monkeypatch.setattr(
        "theseus.data.tokenize.get_tokenizer",
        lambda: pytest.fail("completed restoration should not tokenize"),
    )
    with configuration(cfg):
        restored = StreamJob(spec, base=complete)
        restored(resume=resume)

    store = ObjectStore(spec.hardware)
    restored_state = store.query().node(restored.node).select()[0]
    store.close()
    if resume:
        assert restored.node.seq == complete.seq + 1
    assert restored.node.parent == complete.serialize()
    assert restored_state["data/complete"] is True
    assert restored_state["data/samples"] == 3


def test_fresh_tokenization_rejects_an_existing_materialization(
    tmp_path: Path,
) -> None:
    (tmp_path / "data" / "stream").mkdir(parents=True)
    cfg = build(*StreamJob.config())
    spec = ExecutionSpec.local(str(tmp_path), name="stream")

    with configuration(cfg):
        job = StreamJob(spec)
        with pytest.raises(FileExistsError):
            job()
        job.finish()


@pytest.mark.parametrize("resume", [False, True])
def test_variable_tokenization_continues_from_node_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    monkeypatch.setattr("theseus.data.tokenize.get_tokenizer", Tokenizer)
    path = tmp_path / "data" / "stream"
    path.mkdir(parents=True)
    np.asarray([1, 0], dtype=np.uint32).tofile(path / "train.bin")
    (path / "val.bin").touch()
    cfg = build(*StreamJob.config())
    spec = ExecutionSpec.local(str(tmp_path), name="stream")
    base = Node(name="source", nonce="abcdef")
    store = ObjectStore(spec.hardware)
    store.value(
        base,
        {
            "data/dataset": "stream",
            "data/suffix": "",
            "data/complete": False,
            "data/samples": 1,
            "data/train_tokens": 2,
            "data/val_tokens": 0,
        },
    )
    store.close()

    with configuration(cfg):
        restored = StreamJob(spec, base=base)
        restored(resume=resume)

    np.testing.assert_array_equal(
        np.fromfile(path / "train.bin", dtype=np.uint32),
        np.asarray([1, 0, 2, 0, 3, 0], dtype=np.uint32),
    )
    assert (path / "val.bin").stat().st_size == 0


def test_mapping_config_fields_preserve_values_and_typed_overrides():
    from omegaconf import OmegaConf

    @dataclass
    class MappingConfig:
        counts: dict[str, int] = field("data/task/samples")
        seed: int = field("data/task/seed", default=0)

    config = OmegaConf.merge(
        build(MappingConfig),
        {"data": {"task": {"samples": {"train": 8, "test": 4}, "seed": 10}}},
    )
    with configuration(config):
        args = configure(MappingConfig)
        changed = configure(MappingConfig, counts={"train": 2})
    assert args.counts == {"train": 8, "test": 4}
    assert args.seed == changed.seed == 10
    assert changed.counts == {"train": 2}
    assert config.data.task.samples.train == 8


