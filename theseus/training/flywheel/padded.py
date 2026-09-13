"""
padded.py
Padded Dataset - datasets with pre-padded sequences and masks.
"""

import json
import os
from pathlib import Path

import jax
import numpy as np

from theseus.base.job import ExecutionSpec
from theseus.training.flywheel.dataset import Dataset
from theseus.training.flywheel.io import RowReader


class PaddedDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.cache_train = None
        self.cache_val = None
        self.cache_train_mask = None
        self.cache_val_mask = None
        self.block_size = block_size
        self._reader = RowReader()

        data_dir: Path = spec.hardware.hosts[jax.process_index()].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else data_dir / f"{name}_{suffix}"
        self.path = path

        with open(path / "shape.json", "r") as f:
            self.shape = json.load(f)

        self.has_val = (path / "val.bin").exists()

    def _size(self, split: str) -> int:
        if split == "val" and not self.has_val:
            split = "train"
        return int(self.shape[split][0])

    def _read_rows(self, ix: np.ndarray, split: str) -> dict[str, np.ndarray]:
        """get batches from padded dataset with masks"""

        if not self.has_val and split == "val":
            split = "train"

        shape = tuple(self.shape[split])
        data_dir = self.path
        block_size = self.block_size

        if split == "train":
            if self.cache_train is not None:
                data = self.cache_train
                mask = self.cache_train_mask
            else:
                data = self._reader.map(
                    os.path.join(data_dir, "train.bin"),
                    dtype=np.uint32,
                    shape=shape,
                )
                self.cache_train = data  # type: ignore
                mask = self._reader.map(
                    os.path.join(data_dir, "train.bin.mask"),
                    dtype=np.bool_,
                    shape=shape,
                )
                self.cache_train_mask = mask  # type: ignore
        else:
            if self.cache_val is not None:
                data = self.cache_val
                mask = self.cache_val_mask
            else:
                data = self._reader.map(
                    os.path.join(data_dir, "val.bin"),
                    dtype=np.uint32,
                    shape=shape,
                )
                self.cache_val = data  # type: ignore
                mask = self._reader.map(
                    os.path.join(data_dir, "val.bin.mask"),
                    dtype=np.bool_,
                    shape=shape,
                )
                self.cache_val_mask = mask  # type: ignore

        # check that the dataset is at least as long as the block size
        assert data.shape[1] >= block_size, "Dataset is smaller than block size."
        data = data[:, -(block_size + 1) :]  # type: ignore
        mask = mask[:, -(block_size + 1) :]  # type: ignore

        samples, masks = self._reader.read((data, mask), ix)
        # Allocate the final shape once, including the legacy left-padding
        # case where the stored width is exactly block_size.
        width = samples.shape[1] - 1
        if width == block_size:
            x = samples[:, :-1].astype(np.int64)
            y = samples[:, 1:].astype(np.int64)
            padding_mask = masks[:, :-1].copy()
            y[~masks[:, 1:]] = -1
        else:
            x = np.zeros((len(ix), block_size), dtype=np.int64)
            y = np.full((len(ix), block_size), -1, dtype=np.int64)
            padding_mask = np.zeros((len(ix), block_size), dtype=np.bool_)
            if width:
                x[:, -width:] = samples[:, :-1]
                y[:, -width:] = samples[:, 1:]
                y[:, -width:][~masks[:, 1:]] = -1
                padding_mask[:, -width:] = masks[:, :-1]
        return {"x": x, "y": y, "padding_mask": padding_mask}
