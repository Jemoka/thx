#!/usr/bin/env python3
"""Smoke test for the Muon optimizer wired into gpt/train/pretrain."""

import numpy as np
from pathlib import Path

from theseus.quick import quick
from theseus.training.optimizers import Muon
from theseus.experiments.models.gpt import PretrainGPT

# ---------------------------------------------------------------------------
# Synthetic dataset: create memmap-compatible train.bin / val.bin so the
# MemmapDataset ("Poor Man's Dataloader") has something to read.
# ---------------------------------------------------------------------------

_OUT_ROOT = Path("/tmp/theseus_muon")
_VOCAB_SIZE = 50257  # GPT-2 / tiktoken cl100k range
_NUM_TOKENS = 2**17  # 128K tokens — plenty for any tiny smoke run


def _make_synthetic_data(root: Path) -> None:
    data_dir = root / "data" / "fineweb"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for split in ("train", "val"):
        p = data_dir / f"{split}.bin"
        if not p.exists():
            tokens = rng.integers(1, _VOCAB_SIZE, size=_NUM_TOKENS, dtype=np.uint32)
            tokens.tofile(p)


_OUT_ROOT.mkdir(parents=True, exist_ok=True)
_make_synthetic_data(_OUT_ROOT)


class PretrainGPTMuon(PretrainGPT):
    """GPT pretraining with Muon instead of AdamW."""

    OPTIMIZER = Muon


with quick("/tmp/theseus_muon") as j:
    j.build(PretrainGPTMuon, "smoke_test_muon")
    j.config.architecture.n_layers = 2
    j.config.architecture.n_embd = 128
    j.config.architecture.n_head = 2
    j.config.architecture.block_size = 128

    # Base LR = matrix LR (multiplier=1.0).  Other groups scale from this:
    #   embedding  = lr * 15.0  → 0.30
    #   unembedding = lr * 0.20  → 0.004
    #   scalar      = lr * 25.0  → 0.50
    j.config.optimization.lr = 0.02
    j.config.optimization.weight_decay = 0.1

    # Minimal training run
    j.config.training.batch_size = 8
    j.config.training.per_device_batch_size = 8
    j.config.training.tokens = 8192
    j.config.training.validation_steps = 8
    j.config.training.validation = True
    j.config.training.evaluate = False

    # Log every step, checkpoint never
    j.config.logging.report_interval = 1
    j.config.logging.checkpoint_interval = 100000
    j.config.logging.validation_interval = 2
    j.config.logging.remote = False

    print("Starting Muon smoke test with config:")
    print(
        f"  Model: {j.config.architecture.n_layers} layers, {j.config.architecture.n_embd} embd"
    )
    print(
        f"  Optimizer: muon  (matrix lr={j.config.optimization.lr}, "
        f"embedding lr={j.config.optimization.lr * j.config.optimization.muon.matrix_lr_multiplier:.4f} * "
        f"{j.config.optimization.muon.embedding_lr_multiplier}x)"
    )
    print(
        f"  Training: {j.config.training.tokens} tokens, batch {j.config.training.batch_size}"
    )
    print()

    j()

    print("\nMuon smoke test completed successfully!")
