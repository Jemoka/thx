"""Byte-for-byte reader compatibility, including sparse and overlapping reads."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from theseus.training.flywheel import pmd
from theseus.training.flywheel.padded import PaddedDataset
from theseus.training.flywheel.contrastive import ContrastivePaddedDataset


def spec_at(path):
    return SimpleNamespace(hardware=SimpleNamespace(hosts=[
        SimpleNamespace(cluster=SimpleNamespace(data_dir=path))
    ]))


@pytest.mark.parametrize("width", [8, 9, 17])
@pytest.mark.parametrize("split", ["train", "val"])
def test_padded_reference(tmp_path, width, split):
    path = tmp_path / "data"
    path.mkdir()
    rng = np.random.default_rng(51)
    data = rng.integers(0, 2**32, size=(35, width), dtype=np.uint32)
    mask = rng.random(data.shape) > .3
    data.tofile(path / "train.bin")
    mask.tofile(path / "train.bin.mask")
    (path / "shape.json").write_text(json.dumps({"train": list(data.shape)}))
    ix = np.array([34, 0, 17, 17, 1, 33, 2, 10, 3])
    reader = PaddedDataset(spec_at(tmp_path), 8, "data")
    samples, masks = data[ix, -9:], mask[ix, -9:]
    x, y = samples[:, :-1].astype(np.int64), samples[:, 1:].astype(np.int64)
    y[~masks[:, 1:]] = -1
    padding = 8 - x.shape[1]
    expected = {
        "x": np.pad(x, ((0, 0), (padding, 0))),
        "y": np.pad(y, ((0, 0), (padding, 0)), constant_values=-1),
        "padding_mask": np.pad(masks[:, :-1], ((0, 0), (padding, 0))),
    }
    reader._reader._parallel = True
    actual = reader._read_rows(ix, split)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].flags.c_contiguous
    actual["x"][:] = 0
    assert not np.shares_memory(actual["x"], actual["y"])
    np.testing.assert_array_equal(reader._read_rows(ix, split)["x"], expected["x"])


@pytest.mark.parametrize("block_size", [1, 8, 33, 129])
def test_pmd_sparse_windows_and_spanning_blocks(tmp_path, monkeypatch, block_size):
    monkeypatch.setattr(pmd, "BYTES_PER_BLOCK", 128)
    monkeypatch.setattr(pmd, "BUFFER_BLOCKS", 8)
    path = tmp_path / "data"
    path.mkdir()
    data = np.arange(2003, dtype=np.uint32)
    data.tofile(path / "train.bin")
    reader = pmd.MemmapDataset(spec_at(tmp_path), block_size, "data")
    rng = np.random.default_rng(6)
    for _ in range(25):
        ix = reader._plan("val", rng, 13)
        actual = reader._read_rows(ix, "val")
        samples = np.array([data[i * block_size:(i + 1) * block_size + 1] for i in ix])
        np.testing.assert_array_equal(actual["x"], samples[:, :-1])
        np.testing.assert_array_equal(actual["y"], samples[:, 1:])
        assert actual["padding_mask"].all()
        assert actual["x"].dtype == actual["y"].dtype == np.int64
        assert not np.shares_memory(actual["x"], actual["y"])


def test_pmd_reads_only_needed_blocks_and_reuses_them(tmp_path, monkeypatch):
    monkeypatch.setattr(pmd, "BYTES_PER_BLOCK", 128)
    monkeypatch.setattr(pmd, "BUFFER_BLOCKS", 8)
    path = tmp_path / "data"
    path.mkdir()
    np.arange(1025, dtype=np.uint32).tofile(path / "train.bin")
    reader = pmd.MemmapDataset(spec_at(tmp_path), 8, "data")
    read_ranges = pmd.read_ranges
    bytes_read = []

    def record(source, destination, spans):
        bytes_read.append(sum(source[span].nbytes for span in spans))
        read_ranges(source, destination, spans)

    monkeypatch.setattr(pmd, "read_ranges", record)
    first = reader._read_rows(np.array([0, 1, 0]), "train")
    again = reader._read_rows(np.array([0, 1, 0]), "train")
    assert bytes_read == [128, 0]
    for key in first:
        np.testing.assert_array_equal(first[key], again[key])
    reader._read_rows(np.array([3]), "train")  # shifted label crosses a block
    assert bytes_read[-1] == 128
    assert sum(bytes_read) < pmd.BYTES_PER_BLOCK * pmd.BUFFER_BLOCKS


def test_contrastive_reference_separate_widths_and_validation(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    rng = np.random.default_rng(9)
    shape, arrays = {}, {}
    for split in ("train", "val"):
        shape[split] = {}
        for side, width in (("pos", 8), ("neg", 13)):
            data = rng.integers(0, 2**32, size=(19, width), dtype=np.uint32)
            mask = rng.random(data.shape) > .4
            data.tofile(path / f"{split}.{side}.bin")
            mask.tofile(path / f"{split}.{side}.bin.mask")
            shape[split][side] = list(data.shape)
            arrays[split, side] = data, mask
    (path / "shape.json").write_text(json.dumps(shape))
    reader = ContrastivePaddedDataset(spec_at(tmp_path), 8, "data")
    ix = np.array([18, 0, 0, 9, 1, 17, 3, 2])
    for split in ("train", "val"):
        reader._reader._parallel = True
        actual = reader._read_rows(ix, split)
        for side in ("pos", "neg"):
            data, mask = arrays[split, side]
            expected = data[ix, -8:].astype(np.int64)
            expected[~mask[ix, -8:]] = -1
            np.testing.assert_array_equal(actual[side], expected)
            np.testing.assert_array_equal(actual[f"padding_mask_{side}"], mask[ix, -8:])
            assert actual[side].dtype == np.int64


def test_parallel_padded_short_reads_and_eof(tmp_path, monkeypatch):
    from theseus.training.flywheel import io

    path = tmp_path / "data"
    path.mkdir()
    data = np.arange(12 * 9, dtype=np.uint32).reshape(12, 9)
    data.tofile(path / "train.bin")
    np.ones_like(data, dtype=np.bool_).tofile(path / "train.bin.mask")
    (path / "shape.json").write_text(json.dumps({"train": list(data.shape)}))
    reader = PaddedDataset(spec_at(tmp_path), 8, "data")
    pread = io.os.pread
    monkeypatch.setattr(io.os, "pread", lambda fd, size, offset: pread(fd, min(size, 7), offset))
    indices = np.array([11, 0, 5, 5, 1])
    reader._reader._parallel = True
    actual = reader._read_rows(indices, "train")
    np.testing.assert_array_equal(actual["x"], data[indices, :-1])
    reader._reader._parallel = True
    monkeypatch.setattr(io.os, "pread", lambda *args: b"")
    with pytest.raises(EOFError, match="Dataset ended"):
        reader._read_rows(indices, "train")


@pytest.mark.parametrize("style", ["pmd", "padded", "contrastive"])
def test_batched_zero_retries_match_rowwise_reference(tmp_path, monkeypatch, style):
    import copy
    from theseus.training.flywheel.stream import batches

    path = tmp_path / "data"
    path.mkdir()
    monkeypatch.setattr(pmd, "BYTES_PER_BLOCK", 128)
    monkeypatch.setattr(pmd, "BUFFER_BLOCKS", 8)
    data = np.arange(1000 * 9, dtype=np.uint32).reshape(1000, 9) + 1
    data[:250] = 0
    data.tofile(path / "train.bin")
    np.ones_like(data, dtype=np.bool_).tofile(path / "train.bin.mask")
    (path / "shape.json").write_text(json.dumps({"train": list(data.shape)}))
    if style == "contrastive":
        for side in ("pos", "neg"):
            data.tofile(path / f"train.{side}.bin")
            np.ones_like(data, dtype=np.bool_).tofile(path / f"train.{side}.bin.mask")
        (path / "shape.json").write_text(json.dumps({
            "train": {"pos": list(data.shape), "neg": list(data.shape)}
        }))
    cls = {"pmd": pmd.MemmapDataset, "padded": PaddedDataset,
           "contrastive": ContrastivePaddedDataset}[style]
    actual = cls(spec_at(tmp_path), 8, "data")
    reference = copy.copy(actual)
    if style == "pmd":
        reference._buffers = {}

    def read_rowwise(indices, split, seed=None, offsets=None):
        values = reference._read_rows(indices, split)
        if seed is not None:
            for attempt in range(11):
                x, y = ((values["pos"], values["neg"]) if style == "contrastive"
                        else (values["x"], values["y"]))
                invalid = (x == 0).all(axis=1) & (y == 0).all(axis=1)
                if not invalid.any():
                    break
                if attempt == 10:
                    raise ValueError("Dataset repeatedly returned all-zero samples")
                data_seed, split_seed, start, dataset_index = seed
                for row in np.flatnonzero(invalid):
                    position = start + int(row if offsets is None else offsets[row])
                    rng = np.random.default_rng([data_seed, split_seed, position, dataset_index, attempt])
                    replacement = reference._read_rows(reference._plan(split, rng, 1), split)
                    for key in values:
                        values[key][row] = replacement[key][0]
        return values

    reference._read = read_rowwise
    left = batches([actual, actual], [.6, .4], 23, seed=51)
    right = batches([reference, reference], [.6, .4], 23, seed=51)
    for _ in range(20):
        expected, batch = next(right), next(left)
        for key in expected:
            np.testing.assert_array_equal(batch[key], expected[key])
