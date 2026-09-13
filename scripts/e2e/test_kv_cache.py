"""KV cache parity test.

For each model (GPT, Llama, Qwen, GPTNeoX), verifies that token-by-token
autoregressive generation with KV cache produces identical logits to a
single full-sequence forward pass.

Usage: uv run python scripts/test_kv_cache.py
"""

import sys
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp


def test_kv_cache_parity(
    model_cls: type,
    model_kwargs: dict[str, Any],
    model_name: str,
    seq_len: int = 8,
    vocab_size: int = 128,
    atol: float = 1e-4,
) -> bool:
    """Test that cached autoregressive logits match full-sequence logits."""

    model = model_cls(**model_kwargs)
    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)
    variables = model.init(key, dummy)
    params = variables["params"]

    # Random token sequence
    tokens = jax.random.randint(jax.random.PRNGKey(0), (1, seq_len), 0, vocab_size)

    # --- Full forward pass (no cache) ---
    full_logits, _ = model.apply({"params": params}, tokens, deterministic=True)
    full_logits = np.array(full_logits)  # (1, seq_len, vocab_size)

    # --- Cached autoregressive pass ---
    # Step 1: Prefill with multiple tokens (half the sequence) to test
    # that multi-token prefill correctly sets cache_index to T_prefill.
    prefill_len = seq_len // 2
    prefill_tokens = tokens[:, :prefill_len]  # (1, prefill_len)
    (prefill_logits, _), cache = model.apply(
        {"params": params},
        prefill_tokens,
        deterministic=True,
        mutable=["cache"],
    )

    cached_logits = [np.array(prefill_logits[0, i]) for i in range(prefill_len)]

    # Step 2: Feed remaining tokens one at a time
    for i in range(prefill_len, seq_len):
        next_token = tokens[:, i : i + 1]  # (1, 1)
        (step_logits, _), cache = model.apply(
            {"params": params, **cache},
            next_token,
            deterministic=True,
            mutable=["cache"],
        )
        cached_logits.append(np.array(step_logits[0, 0]))

    cached_logits_arr = np.stack(cached_logits, axis=0)  # (seq_len, vocab_size)

    # --- Compare ---
    max_diff = float(np.max(np.abs(full_logits[0] - cached_logits_arr)))
    mean_diff = float(np.mean(np.abs(full_logits[0] - cached_logits_arr)))

    # Check top-1 agreement at each position
    full_top1 = full_logits[0].argmax(axis=-1)
    cached_top1 = cached_logits_arr.argmax(axis=-1)
    top1_match = int((full_top1 == cached_top1).sum())

    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    print(
        f"  {status} {model_name}: max_diff={max_diff:.8f} mean_diff={mean_diff:.8f} "
        f"top1_match={top1_match}/{seq_len}"
    )
    if not passed:
        # Show per-position diffs for debugging
        for pos in range(seq_len):
            pos_diff = np.max(np.abs(full_logits[0, pos] - cached_logits_arr[pos]))
            print(f"    pos {pos}: max_diff={pos_diff:.8f}")
    return bool(passed)


def main() -> None:
    print("KV Cache Parity Tests")
    print("=" * 60)
    all_passed = True

    # --- GPT (base model) ---
    # GPT uses configure(Block) internally, needs a config context
    from theseus.model.models.base import GPT
    from theseus.config import build, configuration

    gpt_kwargs = dict(
        n_layers=2,
        n_embd=64,
        rope=True,
        block_size=32,
        dropout=0.0,
        vocab_size=128,
    )

    # Build minimal config from GPT's component tree
    cfg = build(*GPT.gather())
    # Override the architecture values
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "architecture.n_layers", 2)
    OmegaConf.update(cfg, "architecture.n_embd", 64)
    OmegaConf.update(cfg, "architecture.n_head", 4)
    OmegaConf.update(cfg, "architecture.rope", True)
    OmegaConf.update(cfg, "architecture.block_size", 32)
    OmegaConf.update(cfg, "architecture.dropout", 0.0)
    OmegaConf.update(cfg, "architecture.vocab_size", 128)
    OmegaConf.update(cfg, "architecture.bias", True)
    OmegaConf.set_struct(cfg, True)

    print("\nGPT (with RoPE):")
    with configuration(cfg):
        # GPT uses bfloat16 internally, so higher tolerance needed
        passed = test_kv_cache_parity(GPT, gpt_kwargs, "GPT+RoPE", atol=2e-3)
    all_passed &= passed

    # --- Llama ---
    from theseus.model.models.contrib.llama import Llama

    llama_kwargs = dict(
        n_layers=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        block_size=32,
        vocab_size=128,
        dropout=0.0,
        attn_dropout=0.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        bias=False,
        attention_bias=False,
    )
    llama_cfg = build(*Llama.gather())
    OmegaConf.set_struct(llama_cfg, False)
    for k, v in llama_kwargs.items():
        OmegaConf.update(llama_cfg, f"architecture.{k}", v)
    OmegaConf.set_struct(llama_cfg, True)

    print("\nLlama:")
    with configuration(llama_cfg):
        passed = test_kv_cache_parity(Llama, llama_kwargs, "Llama")
    all_passed &= passed

    # --- Qwen ---
    from theseus.model.models.contrib.qwen import Qwen

    qwen_kwargs = dict(
        n_layers=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        block_size=32,
        vocab_size=128,
        dropout=0.0,
        attn_dropout=0.0,
        rope_theta=1e6,
        rms_norm_eps=1e-6,
        use_sliding_window=False,
        sliding_window=-1,
        max_window_layers=28,
        bias=False,
    )
    qwen_cfg = build(*Qwen.gather())
    OmegaConf.set_struct(qwen_cfg, False)
    for k, v in qwen_kwargs.items():
        OmegaConf.update(qwen_cfg, f"architecture.{k}", v)
    OmegaConf.set_struct(qwen_cfg, True)

    print("\nQwen:")
    with configuration(qwen_cfg):
        passed = test_kv_cache_parity(Qwen, qwen_kwargs, "Qwen")
    all_passed &= passed

    # --- GPTNeoX ---
    from theseus.model.models.contrib.gpt_neox import GPTNeoX

    neox_kwargs = dict(
        n_layers=2,
        n_embd=64,
        n_head=4,
        n_kv_head=4,
        intermediate_size=128,
        block_size=32,
        vocab_size=128,
        dropout=0.0,
        attn_dropout=0.0,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        layer_norm_eps=1e-5,
        bias=True,
        attention_bias=True,
        use_parallel_residual=True,
    )
    neox_cfg = build(*GPTNeoX.gather())
    OmegaConf.set_struct(neox_cfg, False)
    for k, v in neox_kwargs.items():
        OmegaConf.update(neox_cfg, f"architecture.{k}", v)
    OmegaConf.set_struct(neox_cfg, True)

    print("\nGPTNeoX:")
    with configuration(neox_cfg):
        passed = test_kv_cache_parity(GPTNeoX, neox_kwargs, "GPTNeoX")
    all_passed &= passed

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
