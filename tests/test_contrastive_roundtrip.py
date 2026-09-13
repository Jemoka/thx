"""Round-trip check for contrastive tokenization + loader + async strategy.

Builds a tiny in-memory contrastive dataset, writes memmaps in a temp dir,
loads via ContrastivePaddedDataset and Strategy(Async), and verifies shapes.

Migrated from scripts/test_contrastive_roundtrip.py.
"""

import json
import random
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from theseus.data.tokenize import (
    TokenizeContrastiveDatasetConfig,
    _build_padded_arrays,
    _encode_dataset_item,
)
from theseus.data.tokenizer import TokenizerConfig, get_tokenizer
from theseus.data.datasets import DatasetComponent
from theseus.config import configuration
from omegaconf import OmegaConf
from theseus.training.flywheel.contrastive import ContrastivePaddedDataset
from theseus.training.flywheel.strategy import DatasetStyle, Sampling, Strategy
from theseus.base.job import ExecutionSpec


PAIRS: List[Tuple[str, str]] = [
    ("the quick brown fox", "the slow red fox"),
    ("hello world", "hello mars"),
    ("good code is readable", "good code is obscure"),
    ("unit tests prevent bugs", "unit tests waste time"),
]


class TinyContrastive:
    def __init__(self, data: List[Tuple[str, str]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.data[idx]


def _write_contrastive_data(
    td_path: Path, tokenizer_cfg: TokenizerConfig, label: str
) -> str:
    """Write contrastive memmaps and return dataset name."""
    name = f"tmpcontrast_{label}"
    out = td_path / "data" / name
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(tokenizer_cfg)
    args = TokenizeContrastiveDatasetConfig(
        block_size=32, pad_token=0, val_pct=0.25, seed=42,
    )

    dataset = TinyContrastive(PAIRS)
    indices = list(range(len(dataset)))
    random.seed(args.seed)
    random.shuffle(indices)
    val_size = int(len(dataset) * args.val_pct)
    splits = {"train": indices[val_size:], "val": indices[:val_size]}
    shapes = {}
    dtype = np.uint32

    for split_name, split_indices in splits.items():
        num_samples = len(split_indices)
        pos_tokens = np.memmap(
            out / f"{split_name}.pos.bin", dtype=dtype, mode="w+",
            shape=(num_samples, args.block_size),
        )
        pos_mask = np.memmap(
            out / f"{split_name}.pos.bin.mask", dtype=np.bool_, mode="w+",
            shape=(num_samples, args.block_size),
        )
        neg_tokens = np.memmap(
            out / f"{split_name}.neg.bin", dtype=dtype, mode="w+",
            shape=(num_samples, args.block_size),
        )
        neg_mask = np.memmap(
            out / f"{split_name}.neg.bin.mask", dtype=np.bool_, mode="w+",
            shape=(num_samples, args.block_size),
        )

        for arr_idx, didx in enumerate(split_indices):
            pos_str, neg_str = dataset[didx]
            for s, target_tokens, target_mask in [
                (pos_str, pos_tokens, pos_mask),
                (neg_str, neg_tokens, neg_mask),
            ]:
                ids, mask_list = _encode_dataset_item(s, False, tokenizer, args)
                t, m, *_ = _build_padded_arrays(
                    ids, mask_list, args.block_size, args.pad_token
                )
                target_tokens[arr_idx] = t
                target_mask[arr_idx] = m

        pos_tokens.flush()
        pos_mask.flush()
        neg_tokens.flush()
        neg_mask.flush()
        shapes[split_name] = {
            "pos": [num_samples, args.block_size],
            "neg": [num_samples, args.block_size],
        }

    with open(out / "shape.json", "w") as f:
        json.dump(shapes, f)
    with open(out / "config.json", "w") as f:
        json.dump({}, f)

    return name


class TestContrastiveRoundtrip:
    def test_tiktoken_roundtrip(self):
        with tempfile.TemporaryDirectory() as td, configuration(OmegaConf.create({"data": {"suffix": ""}})):
            td_path = Path(td)
            tok_cfg = TokenizerConfig(backend="tiktoken", name="cl100k_base")
            name = _write_contrastive_data(td_path, tok_cfg, "tiktoken")

            spec = ExecutionSpec.local(root_dir=str(td_path))
            ds = ContrastivePaddedDataset(spec, block_size=32, name=name, suffix="")
            batch = ds.get_batch(batch_size=2, split="train")

            assert batch["padding_mask_pos"].shape == (2, 32)
            assert batch["padding_mask_neg"].shape == (2, 32)

            # Async loader
            strat = Strategy(
                spec, block_size=32,
                mixture=[Sampling(dataset=type("LocalContrastive", (DatasetComponent,), {"DATASET_KEY": name}), rate=1.0, style=DatasetStyle.CONTRASTIVE)],
            )
            async_loader = strat.get_async_batches(batch_size=2, split="train")
            async_batch = async_loader.get_batch()
            async_loader.close()

            assert async_batch["padding_mask_pos"].shape == (2, 32)
            assert async_batch["padding_mask_neg"].shape == (2, 32)
