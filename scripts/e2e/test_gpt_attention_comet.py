#!/usr/bin/env python3
"""Laptop GPT training, checkpoints, and attention plots with real Comet uploads.

Run: JAX_PLATFORMS=cpu uv run python -m scripts.e2e.test_gpt_attention_comet
Uses the already-configured Comet account and the fixtures project. No downloads,
CLI dispatch, or mocked logging: quick owns the normal trainer lifecycle. The
remote API verifies loss metrics and attention figures at the same node steps.
Use --no-remote for a local-only smoke run.
"""

import argparse
import json
from pathlib import Path
import time
from typing import cast

import comet_ml
import jax
import numpy as np

from scripts.smoke.gpt_comet_fixture import TinyGPT
from theseus.analysis import AttentionHeatmapAnalysis
from theseus.quick import quick


class LaptopGPT(TinyGPT):
    ANALYSIS = [AttentionHeatmapAnalysis]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("~/theseus-attention-e2e").expanduser()
    )
    parser.add_argument("--project", default="fixtures")
    parser.add_argument("--name", default="laptop-gpt-attention")
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args()
    assert jax.process_count() == 1 and jax.device_count() == 1, (
        "Run on one local device"
    )
    if not args.no_remote and not comet_ml.config.get_config("comet.api_key"):
        raise RuntimeError("Configure a Comet API key before running this e2e test")
    args.root.mkdir(parents=True, exist_ok=True)

    with quick(args.root) as q:
        q.build(LaptopGPT, args.name, project=args.project, group="laptop")
        q.config.architecture.n_layers = 4
        q.config.architecture.n_embd = 128
        q.config.architecture.n_head = 4
        q.config.architecture.intermediate_size = 512
        q.config.architecture.block_size = 32
        q.config.architecture.vocab_size = 64
        q.config.architecture.dtype.param = "float32"
        q.config.architecture.dtype.activation = "float32"
        q.config.training.batch_size = 1
        q.config.training.per_device_batch_size = 1
        q.config.training.tokens = 256
        q.config.training.validation = False
        q.config.training.evaluate = False
        q.config.training.analyze = True
        q.config.logging.report_interval = 1
        q.config.logging.checkpoint_interval = 4
        q.config.logging.validation_interval = 6
        q.config.logging.remote = not args.no_remote
        q.config.analysis.layer = 0
        q.config.analysis.head = 0
        q.config.analysis.max_tokens = 32
        trainer = q.create()
        experiment_key = comet_ml.get_experiment_key(trainer.node.nonce)
        q()
        rows = (
            q.find().name(trainer.node.name).nonce(trainer.node.nonce).select(raw=True)
        )
        losses = [cast(float, row["train/loss"]) for row in rows if "train/loss" in row]
        checkpoints = [row["_x_seq"] for row in rows if row.get("_x_checkpoint")]
        assert len(losses) == 8 and np.isfinite(losses).all(), losses
        assert losses[-1] < losses[0], losses
        assert checkpoints == [2, 6, 8], checkpoints
        figure = trainer.spec.result_path("attention.pdf")
        assert figure.read_bytes().startswith(b"%PDF"), figure
        report = {
            "root": str(args.root),
            "node": trainer.node.serialize(),
            "parameters": sum(p.size for p in jax.tree.leaves(trainer.state.params)),
            "losses": losses,
            "checkpoint_steps": checkpoints,
            "figure": str(figure),
        }

    if not args.no_remote:
        experiment = comet_ml.API().get_experiment_by_key(experiment_key)
        deadline = time.monotonic() + 60
        while True:
            figures = [
                asset
                for asset in experiment.get_asset_list(asset_type="image", timeout=15)
                if str(asset.get("fileName", "")).startswith("attention")
            ]
            figure_steps = sorted(int(asset["step"]) for asset in figures)
            metrics = experiment.get_metrics("train/loss")
            metric_steps = sorted(int(metric["step"]) for metric in metrics)
            if figure_steps == [2, 8] and metric_steps == list(range(1, 9)):
                break
            if time.monotonic() >= deadline:
                raise AssertionError(
                    {"figure_steps": figure_steps, "metric_steps": metric_steps}
                )
            time.sleep(2)
        assert all(int(asset["fileSize"]) > 0 for asset in figures)
        report.update(comet=experiment.url, attention_steps=figure_steps)

    (args.root / "last-run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
