"""LoRA sharding test with 8 fake CPU devices.

Verifies that after transition_to_lora, LoRA params and optimizer state
are correctly sharded across devices — using the real LoRATrainer init
path, not a hand-built mock.

Must set XLA_FLAGS before importing JAX.
"""

import os
import tempfile

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp

from theseus.config import build, configuration
from theseus.model.models.base import GPT
from theseus.training.lora import (
    LoRATrainer,
    LoRATrainerConfig,
    LoRATrainState,
)


class _TestableLoRATrainer(LoRATrainer):
    """LoRATrainer subclass that skips data loading, wandb, and evals.

    Everything else (topology, model init, state init, batch config)
    runs through the real code path.  Prefixed with _ so pytest doesn't
    try to collect it as a test class.
    """

    MODEL = GPT
    CONFIG = LoRATrainerConfig
    DATASET = []

    def _init_data(self, spec):
        # Skip — needs tokenized data on disk
        self._in_lora_phase = False
        self._pre_lora_total = sum(self.args.pre_lora_tokens)
        self._current_stage = 0
        self._pre_segment_ends = []
        cumulative = 0
        for t in self.args.pre_lora_tokens:
            cumulative += t
            self._pre_segment_ends.append(cumulative)
        self._post_segment_ends = []
        cumulative = self._pre_lora_total
        for t in self.args.post_lora_tokens:
            cumulative += t
            self._post_segment_ends.append(cumulative)

    def _init_wandb(self, spec):
        pass

    def _init_counters_and_eval(self):
        self.best_val_score_ = float("-inf")

    def evaluator(self):
        return None


class TestLoRASharding:
    def _make_cfg(self):
        from omegaconf import OmegaConf

        cfg = build(*_TestableLoRATrainer.config())
        OmegaConf.set_struct(cfg, False)
        cfg.architecture.n_layers = 2
        cfg.architecture.n_embd = 64
        cfg.architecture.n_head = 8
        cfg.architecture.block_size = 32
        cfg.architecture.vocab_size = 128
        cfg.architecture.dropout = 0.0
        cfg.architecture.rope = True
        cfg.architecture.bias = True
        cfg.architecture.dtype = OmegaConf.create(
            {"param": "float32", "activation": "float32"}
        )
        cfg.training.batch_size = 8
        cfg.training.per_device_batch_size = 1
        cfg.training.pre_lora_tokens = [512]
        cfg.training.post_lora_tokens = [512]
        # rank must be >= the TP count (8) so the LoRA matrices are large
        # enough to shard across devices without remainder errors.
        cfg.optimization.lora.rank = 8
        cfg.optimization.lora.alpha = 8.0
        cfg.optimization.lora.target_modules = ["kernel"]
        cfg.logging.remote = False
        cfg.logging.checkpoint_interval = 999999
        cfg.logging.validation_interval = 999999
        OmegaConf.set_struct(cfg, True)
        return cfg

    def _make_trainer(self):
        """Create trainer inside a config context that stays alive."""
        from theseus.base.job import ExecutionSpec

        cfg = self._make_cfg()
        # Store config context manager so it stays alive for the test
        self._cfg_ctx = configuration(cfg)
        self._cfg_ctx.__enter__()

        tmpdir = tempfile.mkdtemp()
        spec = ExecutionSpec.local(root_dir=tmpdir, name="test_sharding")
        self._trainer = _TestableLoRATrainer(spec)
        self._trainer.setup()
        return self._trainer

    def _cleanup_cfg(self):
        if hasattr(self, "_trainer"):
            self._trainer.finish()
        if hasattr(self, "_cfg_ctx"):
            self._cfg_ctx.__exit__(None, None, None)

    def test_pre_lora_state_sharded(self):
        """Initial state (pre-LoRA) should be sharded across 8 devices."""
        trainer = self._make_trainer()
        try:
            assert not trainer._in_lora_phase
            for leaf in jax.tree_util.tree_leaves(trainer.state.params):
                if hasattr(leaf, "sharding") and leaf.size > 1:
                    assert len(leaf.sharding.device_set) > 1, (
                        f"Pre-LoRA param {leaf.shape} on only "
                        f"{len(leaf.sharding.device_set)} device(s)"
                    )
        finally:
            self._cleanup_cfg()

    def test_lora_state_sharded_after_transition(self):
        """After _transition_to_lora, LoRA params + optimizer are sharded."""
        trainer = self._make_trainer()
        try:
            trainer._transition_to_lora()

            assert trainer._in_lora_phase
            assert isinstance(trainer.state, LoRATrainState)

            # LoRA params sharded
            for leaf in jax.tree_util.tree_leaves(trainer.state.params):
                if leaf is not None and hasattr(leaf, "sharding") and leaf.size > 1:
                    assert len(leaf.sharding.device_set) > 1, (
                        f"LoRA param {leaf.shape} on only "
                        f"{len(leaf.sharding.device_set)} device(s)"
                    )

            # base_params sharded
            for leaf in jax.tree_util.tree_leaves(trainer.state.base_params):
                if hasattr(leaf, "sharding") and leaf.size > 1:
                    assert len(leaf.sharding.device_set) > 1, (
                        f"Base param {leaf.shape} on only "
                        f"{len(leaf.sharding.device_set)} device(s)"
                    )

            # Optimizer state sharded
            opt_leaves = jax.tree_util.tree_leaves(trainer.state.opt_state)
            assert len(opt_leaves) > 0
            for leaf in opt_leaves:
                if (
                    hasattr(leaf, "sharding")
                    and hasattr(leaf, "shape")
                    and leaf.size > 1
                ):
                    assert len(leaf.sharding.device_set) > 1, (
                        f"Opt state {leaf.shape} on only "
                        f"{len(leaf.sharding.device_set)} device(s)"
                    )
        finally:
            self._cleanup_cfg()

    def test_sharded_forward_matches_base(self):
        """Sharded LoRA forward with B=0 should match base model output."""
        trainer = self._make_trainer()
        try:
            dummy = jnp.zeros((1, 8), dtype=jnp.int32)
            logits_base, _ = trainer.model.apply(
                {"params": trainer.state.params}, dummy, deterministic=True
            )

            trainer._transition_to_lora()

            batch = {
                "x": dummy,
                "y": jnp.ones_like(dummy),
                "padding_mask": jnp.ones_like(dummy, dtype=jnp.bool_),
            }
            logits_lora, _, _ = LoRATrainer.forward(
                trainer.state, trainer.state.params, batch, deterministic=True
            )

            # bf16 cast on base_params introduces ~1e-3 error
            assert jnp.allclose(logits_base, logits_lora, atol=5e-3), (
                f"max diff = {float(jnp.max(jnp.abs(logits_base - logits_lora)))}"
            )
        finally:
            self._cleanup_cfg()
