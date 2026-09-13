#!/usr/bin/env python3
"""
Test PerplexityEvaluation end-to-end using TinyStoriesEval on a randomly
initialized model via the quick runner.
"""

from theseus.quick import quick
from theseus.experiments.models.gpt import PretrainGPT
from theseus.evaluation.datasets.tinystories import TinyStoriesEval


class TinyStoriesTrainer(PretrainGPT):
    EVALUATION = [TinyStoriesEval]


with quick() as j:
    j.build(TinyStoriesTrainer, "test_tinystories_eval")
    j.config.architecture.n_layers = 2
    j.config.architecture.n_embd = 64
    j.config.architecture.n_head = 2
    j.config.architecture.block_size = 512  # long enough for TinyStories

    # Minimal training config (just enough to init the trainer)
    j.config.training.batch_size = 8
    j.config.training.per_device_batch_size = 8
    j.config.training.tokens = 8192
    j.config.training.validation = False
    j.config.training.evaluate = True

    j.config.logging.remote = False
    j.config.logging.checkpoint_interval = 100000
    j.config.logging.report_interval = 1
    j.config.logging.validation_interval = 100000

    # Create the trainer (initializes model, state, mesh, evaluator)
    trainer = j.create()

    print("Running TinyStories evaluation on randomly initialized model...")
    results = trainer.inference.evaluate()

    ppl = results["tinystories_ppl"]
    print(f"\nTinyStories ppl : {ppl:.2f}")

    # A random model should have high perplexity
    assert ppl > 1, "ppl should be > 1 for any reasonable model"
    print("\nTest passed!")
