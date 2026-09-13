"""
contrastive.py
Padded contrastive dataset: paired positive/negative sequences with masks.
"""

import json
import os
from pathlib import Path
from typing import Tuple

import jax
import numpy as np

from theseus.base.job import ExecutionSpec
from theseus.training.flywheel.dataset import Dataset
from theseus.training.flywheel.io import RowReader


class ContrastivePaddedDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.block_size = block_size
        self._reader = RowReader()

        data_dir: Path = spec.hardware.hosts[jax.process_index()].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else data_dir / f"{name}_{suffix}"
        self.path = path

        with open(path / "shape.json", "r") as f:
            self.shape = json.load(f)

        # caches
        self.cache = {
            "train_pos": None,
            "train_neg": None,
            "val_pos": None,
            "val_neg": None,
            "train_pos_mask": None,
            "train_neg_mask": None,
            "val_pos_mask": None,
            "val_neg_mask": None,
        }

        self.has_val = (path / "val.pos.bin").exists()

    def _load_split(
        self, split: str
    ) -> Tuple[np.memmap, np.memmap, np.memmap, np.memmap, tuple[int, int]]:
        if not self.has_val and split == "val":
            split = "train"

        shape_pos = tuple(self.shape[split]["pos"])
        shape_neg = tuple(self.shape[split]["neg"])
        data_dir = self.path

        if split == "train":
            cache_prefix = "train"
        else:
            cache_prefix = "val"

        pos_key = f"{cache_prefix}_pos"
        neg_key = f"{cache_prefix}_neg"
        pos_mask_key = f"{cache_prefix}_pos_mask"
        neg_mask_key = f"{cache_prefix}_neg_mask"

        if self.cache[pos_key] is None:
            pos = self._reader.map(
                os.path.join(data_dir, f"{split}.pos.bin"),
                dtype=np.uint32,
                shape=shape_pos,
            )
            pos_mask = self._reader.map(
                os.path.join(data_dir, f"{split}.pos.bin.mask"),
                dtype=np.bool_,
                shape=shape_pos,
            )
            neg = self._reader.map(
                os.path.join(data_dir, f"{split}.neg.bin"),
                dtype=np.uint32,
                shape=shape_neg,
            )
            neg_mask = self._reader.map(
                os.path.join(data_dir, f"{split}.neg.bin.mask"),
                dtype=np.bool_,
                shape=shape_neg,
            )

            self.cache[pos_key] = pos  # type: ignore
            self.cache[neg_key] = neg  # type: ignore
            self.cache[pos_mask_key] = pos_mask  # type: ignore
            self.cache[neg_mask_key] = neg_mask  # type: ignore
        else:
            pos = self.cache[pos_key]  # type: ignore
            neg = self.cache[neg_key]  # type: ignore
            pos_mask = self.cache[pos_mask_key]  # type: ignore
            neg_mask = self.cache[neg_mask_key]  # type: ignore

        return pos, neg, pos_mask, neg_mask, shape_pos

    def _size(self, split: str) -> int:
        if split == "val" and not self.has_val:
            split = "train"
        return int(self.shape[split]["pos"][0])

    def _read_rows(self, ix: np.ndarray, split: str) -> dict[str, np.ndarray]:
        """Get batches from contrastive padded dataset with masks."""

        if not self.has_val and split == "val":
            split = "train"

        pos, neg, pos_mask_full, neg_mask_full, shape_pos = self._load_split(split)
        block_size = self.block_size

        # check that the dataset is at least as long as the block size
        assert pos.shape[1] >= block_size, "Dataset is smaller than block size."
        assert neg.shape[1] >= block_size, "Dataset is smaller than block size."

        pos_data = pos[:, -block_size:]
        neg_data = neg[:, -block_size:]
        pos_mask = pos_mask_full[:, -block_size:]
        neg_mask = neg_mask_full[:, -block_size:]

        x_pos, y_neg, mask_pos, mask_neg = self._reader.read(
            (pos_data, neg_data, pos_mask, neg_mask), ix
        )

        x_pos = x_pos.astype(np.int64)
        y_neg = y_neg.astype(np.int64)

        # set padding tokens to -1 so model.loss() masks them correctly
        x_pos[~mask_pos] = -1
        y_neg[~mask_neg] = -1

        # pad up to block size if needed (should already be exact, but keep safety)
        def pad_arr(
            arr: np.ndarray, m_arr: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if arr.shape[-1] < block_size:
                pad_len = block_size - arr.shape[-1]
                arr = np.concatenate(
                    (np.full((arr.shape[0], pad_len), -1, dtype=np.int64), arr),
                    axis=-1,
                )
                m_arr = np.concatenate(
                    (
                        np.full((m_arr.shape[0], pad_len), False, dtype=np.bool_),
                        m_arr,
                    ),
                    axis=-1,
                )
            return arr, m_arr

        x_pos, mask_pos = pad_arr(x_pos, mask_pos)
        y_neg, mask_neg = pad_arr(y_neg, mask_neg)

        return {
            "pos": x_pos,
            "neg": y_neg,
            "padding_mask_pos": mask_pos,
            "padding_mask_neg": mask_neg,
        }
