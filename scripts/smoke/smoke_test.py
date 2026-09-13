#!/usr/bin/env python3
"""Smoke test for refactored training code."""

from theseus.quick import quick

with quick() as j:
    j.build("gpt/train/pretrain", "smoke_test")
    j.config.architecture.n_layers = 2
    j.config.architecture.n_embd = 128
    j.config.architecture.n_head = 2
    j.config.architecture.block_size = 128

    # Minimal training run
    j.config.training.batch_size = 8
    j.config.training.per_device_batch_size = 8
    j.config.training.tokens = 8192  # ~1000 steps
    j.config.training.validation_steps = 8
    j.config.training.validation = True
    j.config.training.evaluate = False

    # Log every step, checkpoint never
    j.config.logging.report_interval = 1
    j.config.logging.checkpoint_interval = 100000
    j.config.logging.validation_interval = 2
    j.config.logging.remote = False

    print("Starting smoke test with config:")
    print(
        f"  Model: {j.config.architecture.n_layers} layers, {j.config.architecture.n_embd} embd"
    )
    print(f"  Training: {j.config.training.tokens} tokens")
    print(f"  Batch size: {j.config.training.batch_size}")
    print()

    j()

    print("\nSmoke test completed successfully!")
