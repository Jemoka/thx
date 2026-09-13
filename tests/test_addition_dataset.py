from omegaconf import OmegaConf

from theseus.config import build, configuration
from theseus.data.datasets.addition import Addition
from theseus.data.tokenizer import TokenizerConfig, get_tokenizer
from theseus.evaluation.datasets.addition import AdditionEval, AdditionEvalConfig


ADDITION_CONFIG = {
    "samples": {"train": 8, "test": 4},
    "seeds": {"train": 10, "test": 20},
    "left_min": 100,
    "left_max": 999,
    "right_min": 100,
    "right_max": 999,
    "operand_width": 3,
    "answer_width": 4,
    "tokens": {
        "eos": 0,
        "bos": 1,
        "middle": 2,
        "plus": 3,
        "equals": 4,
        "separator": 5,
        "digit_offset": 6,
    },
}


def test_addition_is_deterministic_trivial_token_text() -> None:
    config = OmegaConf.merge(
        build(*Addition.config()),
        {"data": {"addition": {**ADDITION_CONFIG, "split": "test"}}},
    )
    with configuration(config):
        dataset = Addition()
        first = dataset[0]
        second = Addition()[0]
    token_ids = [int(token_id) for token_id in first.split()]

    assert first == second
    assert len(token_ids) == 15
    assert token_ids[0] == dataset.bos_token
    assert token_ids[-1] == dataset.eos_token
    assert dataset.deserialize(token_ids).startswith("<bos> ")
    assert " + " in dataset.deserialize(token_ids)
    assert " = <mid> " in dataset.deserialize(token_ids)
    assert dataset.deserialize(token_ids).endswith(" <eos>")

    config = build(TokenizerConfig)
    config.tokenizer.backend = "trivial"
    config.tokenizer.name = "trivial"
    with configuration(config):
        assert get_tokenizer().encode(first) == token_ids


def test_addition_evaluation_uses_numeric_dataset_boundary() -> None:
    config = OmegaConf.merge(
        build(*AdditionEval.config()), {"data": {"addition": ADDITION_CONFIG}}
    )

    with configuration(config):
        evaluation = AdditionEval()
        prompt, answer = evaluation.get(0)

    assert len(evaluation) == 4
    assert prompt.endswith(f" {evaluation.dataset.middle_token}")
    assert not answer.endswith(f" {evaluation.dataset.eos_token}")
    assert evaluation.dataset[0] == (
        f"{prompt} {answer} {evaluation.dataset.eos_token}"
    )
    assert evaluation.check(answer, answer)


def test_addition_materializes_with_static_tokenization_job(tmp_path):
    import numpy as np
    from theseus.base import ExecutionSpec
    from theseus.experiments.data.blockwise import TokenizeAddition
    from theseus.training.flywheel.padded import PaddedDataset

    spec = ExecutionSpec.local(str(tmp_path), name="addition")
    config = OmegaConf.merge(
        build(*TokenizeAddition.config()),
        {
            "data": {"addition": ADDITION_CONFIG, "val_pct": 0.25},
            "architecture": {"block_size": 16},
            "tokenizer": {"backend": "trivial", "name": "trivial"},
        },
    )
    with configuration(config):
        job = TokenizeAddition(spec)
        job()
        expected = {tuple(get_tokenizer().encode(Addition()[i])) for i in range(8)}
    reader = PaddedDataset(spec, 15, "addition")
    data = reader._read_rows(np.arange(reader._size("train")), "train")
    assert data["x"].shape[1] == 15
    for row, mask in zip(data["x"], data["padding_mask"]):
        # The padded source ends in EOS; shifted inputs omit that final token.
        assert tuple(row[mask]) + (0,) in expected


def test_tokenizer_can_deduplicate_before_its_normal_split(tmp_path, monkeypatch):
    import json
    import numpy as np
    from theseus.base import ExecutionSpec
    from theseus.experiments.data.blockwise import TokenizeAddition

    monkeypatch.setattr(
        Addition, "__getitem__", lambda self, index: f"1 {6 + index % 3} 0"
    )
    for replacement, expected in [(True, 8), (False, 3)]:
        (tmp_path / str(replacement)).mkdir()
        spec = ExecutionSpec.local(str(tmp_path / str(replacement)), name="duplicates")
        cfg = OmegaConf.merge(
            build(*TokenizeAddition.config()),
            {
                "data": {
                    "addition": ADDITION_CONFIG,
                    "val_pct": 0.34,
                    "tokenize": {"replacement": replacement},
                },
                "architecture": {"block_size": 4},
                "tokenizer": {"backend": "trivial", "name": "trivial"},
            },
        )
        with configuration(cfg):
            instance = TokenizeAddition(spec)
            instance()
            output = instance.output_path
        shape = json.loads((output / "shape.json").read_text())
        assert sum(count for count, _ in shape.values()) == expected
        if not replacement:
            train = np.fromfile(
                output / "train.bin", dtype=np.uint32
            ).reshape(-1, 4)
            val = np.fromfile(
                output / "val.bin", dtype=np.uint32
            ).reshape(-1, 4)
            assert not set(map(tuple, train)) & set(map(tuple, val))
