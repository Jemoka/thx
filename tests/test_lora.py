"""Tests for LoRA parameter utilities, model compatibility, and train state.

Inspired by scripts/llama_parity.py — tests LoRA injection, merging,
forward pass, and critically: that the optimizer actually updates LoRA
params (not frozen base) during a training step.
"""

import pytest
import jax
import jax.numpy as jnp
import optax
from contextlib import contextmanager
from typing import Any

from theseus.config import build, configuration, configure
from theseus.training.lora import (
    LoRATrainState,
    param_filter,
    inject_lora_params,
    merge_lora_params,
    count_lora_params,
)


# ---------------------------------------------------------------------------
# Config context helpers
# ---------------------------------------------------------------------------


@contextmanager
def _gpt_config_ctx():
    from theseus.model.models.base import GPT
    from theseus.model.block import Block
    from theseus.model.layers import LayerNorm
    from theseus.model.layers.mlp import MLP
    from theseus.model.attention import SelfAttention, RopeAttention

    cfg = build(GPT, Block, LayerNorm, MLP, SelfAttention, RopeAttention)
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 64
    cfg.architecture.n_head = 4
    cfg.architecture.block_size = 32
    cfg.architecture.vocab_size = 128
    cfg.architecture.dropout = 0.0
    cfg.architecture.rope = True
    cfg.architecture.bias = True
    cfg.architecture.dtype = OmegaConf.create(
        {"param": "float32", "activation": "float32"}
    )
    OmegaConf.set_struct(cfg, True)
    with configuration(cfg):
        yield




# ---------------------------------------------------------------------------
# Core LoRA utilities
# ---------------------------------------------------------------------------


class TestParamFilter:
    def test_filter_kernel(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            mask_leaves = jax.tree_util.tree_leaves(mask)
            assert any(mask_leaves), "No kernels found in GPT params"
            assert not all(mask_leaves), "All params are kernels? Unexpected."

    def test_filter_empty(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, [])
            assert not any(jax.tree_util.tree_leaves(mask))


class TestInjectLoRA:
    def test_shapes(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            rank = 4
            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                params, mask, rank, jax.random.PRNGKey(1)
            )

            def _check(p: Any, m: Any, a: Any, b: Any) -> Any:
                if m and p.ndim == 2:
                    assert a is not None
                    assert a.shape == (p.shape[0], rank)
                    assert b is not None
                    assert b.shape == (rank, p.shape[1])
                return p

            jax.tree_util.tree_map(
                _check, params, mask, lora_A, lora_B, is_leaf=lambda x: x is None
            )

    def test_b_initialized_zero(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            _, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))

            for leaf in jax.tree_util.tree_leaves(lora_B):
                if leaf is not None and hasattr(leaf, "shape"):
                    assert jnp.allclose(leaf, 0.0)


class TestMergeLoRA:
    def test_zero_b_preserves_base(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                params, mask, 4, jax.random.PRNGKey(1)
            )
            merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)

            for base_leaf, merged_leaf in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(merged),
            ):
                assert jnp.allclose(base_leaf, merged_leaf, atol=1e-6)

    def test_nonzero_delta(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                params, mask, 4, jax.random.PRNGKey(1)
            )

            lora_B_nz = jax.tree_util.tree_map(
                lambda b: jnp.ones_like(b) if b is not None else None,
                lora_B,
                is_leaf=lambda x: x is None,
            )
            merged = merge_lora_params(params, lora_A, lora_B_nz, alpha=4.0, rank=4)

            any_different = False
            for base_leaf, merged_leaf in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(merged),
            ):
                if not jnp.allclose(base_leaf, merged_leaf, atol=1e-6):
                    any_different = True
                    break
            assert any_different


class TestCountLoRA:
    def test_count(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            rank = 4
            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                params, mask, rank, jax.random.PRNGKey(1)
            )
            count = count_lora_params(lora_A, lora_B)
            assert count > 0


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestLoRAForwardGPT:
    def test_forward_matches_base_with_zero_b(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            idx = jnp.zeros((1, 8), dtype=jnp.int32)
            params = model.init(jax.random.PRNGKey(0), idx)["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                params, mask, 4, jax.random.PRNGKey(1)
            )
            merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)

            logits_base, _ = model.apply({"params": params}, idx, deterministic=True)
            logits_lora, _ = model.apply({"params": merged}, idx, deterministic=True)
            assert jnp.allclose(logits_base, logits_lora, atol=1e-5)

    def test_gradients_flow_through_lora(self):
        """Verify gradients w.r.t. LoRA B params are non-zero."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            idx = jnp.zeros((1, 8), dtype=jnp.int32)
            targets = jnp.ones((1, 8), dtype=jnp.int32)
            params = model.init(jax.random.PRNGKey(0), idx)["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                params, mask, 4, jax.random.PRNGKey(1)
            )

            # Set B to small non-zero so gradients are meaningful
            lora_B_nz = jax.tree_util.tree_map(
                lambda b: jnp.ones_like(b) * 0.01 if b is not None else None,
                lora_B,
                is_leaf=lambda x: x is None,
            )

            def loss_fn(b: Any) -> jax.Array:
                merged = merge_lora_params(params, lora_A, b, alpha=4.0, rank=4)
                _, loss = model.apply(
                    {"params": merged}, idx, targets=targets, deterministic=True
                )
                return loss

            grad = jax.grad(loss_fn, allow_int=True)(lora_B_nz)

            has_grad = False
            for leaf in jax.tree_util.tree_leaves(grad):
                if leaf is not None and hasattr(leaf, "shape") and leaf.size > 0:
                    if jnp.any(leaf != 0):
                        has_grad = True
                        break
            assert has_grad




# ---------------------------------------------------------------------------
# LoRATrainState integration — exercises the real forward/grad/apply path
# ---------------------------------------------------------------------------


def _make_lora_state(model, full_params, rank=4, lr=1e-3):
    """Create a LoRATrainState the same way the trainer does."""
    mask = param_filter(full_params, ["kernel"])
    lora_A, lora_B = inject_lora_params(full_params, mask, rank, jax.random.PRNGKey(1))

    # Non-zero B so gradients flow from step 0
    lora_B = jax.tree_util.tree_map(
        lambda b: jnp.ones_like(b) * 0.01 if b is not None else None,
        lora_B,
        is_leaf=lambda x: x is None,
    )

    base_params = jax.tree_util.tree_map(lambda x: x.copy(), full_params)
    lora_params = {"lora_A": lora_A, "lora_B": lora_B}

    return LoRATrainState.create(  # type: ignore[no-untyped-call]
        apply_fn=model.apply,
        params=lora_params,
        base_params=base_params,
        tx=optax.adam(lr),
        lora_alpha=float(rank),
        lora_rank=rank,
    )


def _make_batch(idx, targets):
    """Build a batch dict matching what forward() expects."""
    return {
        "x": idx,
        "y": targets,
        "padding_mask": jnp.ones_like(idx, dtype=jnp.bool_),
    }


def _lora_train_step(forward_fn, state, batch):
    """One train step using the trainer's static forward — same path as
    BaseTrainer.train_step but without accumulation/scan overhead."""

    def loss_fn(params):
        _logits, loss, _meta = forward_fn(
            state, params, batch, key=jax.random.PRNGKey(0), deterministic=True
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)  # type: ignore[no-untyped-call]
    return state, loss


class TestLoRATrainState:
    """Full-cycle integration: create state -> forward -> grad -> apply ->
    verify LoRA changed, base frozen, model output changed, base-only
    forward unchanged."""

    def test_full_cycle(self):
        from theseus.training.lora import LoRATrainer

        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            idx = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
            targets = jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=jnp.int32)
            full_params = model.init(jax.random.PRNGKey(0), idx)["params"]
            batch = _make_batch(idx, targets)

            state = _make_lora_state(model, full_params, rank=4, lr=1e-2)

            # Snapshot base and lora before training
            base_before = jax.tree_util.tree_map(lambda x: x.copy(), state.base_params)
            lora_before = jax.tree_util.tree_map(
                lambda x: x.copy() if x is not None else None,
                state.params,
                is_leaf=lambda x: x is None,
            )

            # Forward before step (through trainer's forward)
            logits_before, loss_before, _ = LoRATrainer.forward(
                state, state.params, batch, deterministic=True
            )

            # Bare model forward with base params only (no LoRA)
            logits_base_only, _ = model.apply(
                {"params": full_params}, idx, deterministic=True
            )

            # --- Train step via trainer's forward ---
            state, _ = _lora_train_step(LoRATrainer.forward, state, batch)

            # 1) base_params must be unchanged
            for b, a in zip(
                jax.tree_util.tree_leaves(base_before),
                jax.tree_util.tree_leaves(state.base_params),
            ):
                assert jnp.array_equal(b, a), "base_params was modified!"

            # 2) LoRA params must have changed
            lora_changed = False
            for b, a in zip(
                jax.tree_util.tree_leaves(lora_before),
                jax.tree_util.tree_leaves(state.params),
            ):
                if b is not None and a is not None and hasattr(b, "shape"):
                    if not jnp.array_equal(b, a):
                        lora_changed = True
                        break
            assert lora_changed, "LoRA params were NOT updated!"

            # 3) Merged forward output must differ after step
            logits_after, _, _ = LoRATrainer.forward(
                state, state.params, batch, deterministic=True
            )
            assert not jnp.allclose(logits_before, logits_after, atol=1e-6), (
                "Model output didn't change after training step!"
            )

            # 4) Base-only forward (no LoRA) must be unchanged
            logits_base_after, _ = model.apply(
                {"params": full_params}, idx, deterministic=True
            )
            assert jnp.allclose(logits_base_only, logits_base_after, atol=1e-7), (
                "Base model output changed — LoRA leaked into base params!"
            )


# ---------------------------------------------------------------------------
# Contrib model tests
# ---------------------------------------------------------------------------


class TestLoRAContribModels:
    def _test_model_lora(self, model: Any, idx: Any) -> None:
        params = model.init(jax.random.PRNGKey(0), idx)["params"]

        mask = param_filter(params, ["kernel"])
        assert any(jax.tree_util.tree_leaves(mask)), "No kernels found"

        lora_A, lora_B = inject_lora_params(
            params, mask, 4, jax.random.PRNGKey(1)
        )
        assert count_lora_params(lora_A, lora_B) > 0

        # Zero B merge preserves output
        merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)
        logits_base, _ = model.apply({"params": params}, idx, deterministic=True)
        logits_lora, _ = model.apply({"params": merged}, idx, deterministic=True)
        assert jnp.allclose(logits_base, logits_lora, atol=1e-4)

    def _contrib_config_ctx(self, model_cls: Any, **overrides: Any) -> Any:
        from omegaconf import OmegaConf

        all_types = model_cls.gather()
        cfg = build(*all_types)
        OmegaConf.set_struct(cfg, False)
        defaults = dict(
            n_layers=2, n_embd=64, n_head=4, n_kv_head=4,
            block_size=32, vocab_size=128, dropout=0.0, attn_dropout=0.0,
            intermediate_size=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            layer_norm_eps=1e-5, bias=False, attention_bias=False,
            use_sliding_window=False, sliding_window=32, max_window_layers=2,
            use_parallel_residual=True, partial_rotary_factor=1.0,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            try:
                setattr(cfg.architecture, k, v)
            except Exception:
                pass
        cfg.architecture.dtype = OmegaConf.create(
            {"param": "float32", "activation": "float32"}
        )
        OmegaConf.set_struct(cfg, True)
        return configuration(cfg)

    def test_llama(self):
        from theseus.model.models.contrib.llama import Llama

        with self._contrib_config_ctx(Llama):
            model = configure(Llama)
            self._test_model_lora(model, jnp.zeros((1, 8), dtype=jnp.int32))

    def test_gpt_neox(self):
        from theseus.model.models.contrib.gpt_neox import GPTNeoX

        with self._contrib_config_ctx(GPTNeoX):
            model = configure(GPTNeoX)
            self._test_model_lora(model, jnp.zeros((1, 8), dtype=jnp.int32))
