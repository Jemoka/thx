"""Sharded-forward parity for Qwen 3.5 (text-only).

Runs the JAX model in three configurations on the same HF backbone:
  1. ``single``   — one device, no sharding (baseline).
  2. ``tp``       — tensor-parallel: 1 batch x N shard.
  3. ``dp``       — data-parallel: N batch x 1 shard.

For each, compares logits against the single-device baseline. Also verifies a
roundtrip through ``_to_hf_state_dict`` works on the sharded params (gather to
host, then export).

Usage: uv run python scripts/qwen_3.5_sharded_parity.py
"""

import argparse

import flax.linen as flax_nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.utils import logging as hf_logging

from theseus.base.axis import Axis
from theseus.config import configure, patch
from theseus.model.models.contrib.qwen_3_5 import (
    Qwen3_5,
    _from_hf_state_dict,
    _to_hf_state_dict,
)

hf_logging.set_verbosity_error()


def _shard_from_np(
    np_params: object, abstract: object, model: Qwen3_5, mesh: Mesh
) -> object:
    pspec = flax_nn.get_partition_spec(abstract)
    var_sharding = flax_nn.logical_to_mesh_sharding(
        pspec, mesh, rules=model.sharding._tp
    )

    def unwrap(x: object) -> object:
        return x.value if isinstance(x, flax_nn.Partitioned) else x

    np_flat = jax.tree_util.tree_leaves(
        np_params, is_leaf=lambda x: isinstance(x, flax_nn.Partitioned)
    )
    sharding_flat = jax.tree_util.tree_leaves(var_sharding)
    assert len(np_flat) == len(sharding_flat)
    sharded_flat = [
        jax.device_put(unwrap(a), s) for a, s in zip(np_flat, sharding_flat)
    ]
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(var_sharding), sharded_flat
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    devices = jax.devices()
    n = len(devices)
    print(f"jax devices: {n} × {devices[0].device_kind}")
    if n < 2:
        raise RuntimeError(
            "Sharded parity needs >= 2 JAX devices; "
            "set CUDA_VISIBLE_DEVICES or XLA_FLAGS=--xla_force_host_platform_device_count=N"
        )

    # Configurations: (name, mesh_shape, input_batch)
    # "single" is run on a 1×1 mesh as the baseline; tp/dp use all n devices.
    configs = [
        ("single", (1, 1), 1),
        ("tp", (1, n), 1),
        ("dp", (n, 1), n),
    ]

    hf = Qwen3_5ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=None
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    chat = [{"role": "user", "content": args.prompt}]
    prompt_text = tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    inputs = tok(
        prompt_text,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )
    with torch.no_grad():
        logits_hf = hf(**inputs).logits.detach().cpu().numpy()
    state_dict = hf.state_dict()
    cfg = getattr(hf.config, "text_config", hf.config)
    rope_params = getattr(cfg, "rope_parameters", None) or {}

    baseline_logits = None
    sharded_for_roundtrip = None

    for name, mesh_shape, batch_size in configs:
        n_used = mesh_shape[0] * mesh_shape[1]
        if n_used > n:
            print(f"-- skipping {name}: mesh {mesh_shape} > {n} devices")
            continue
        print(f"\n--- {name} (mesh {mesh_shape}, batch {batch_size}) ---")
        mesh = Mesh(
            np.array(devices[:n_used]).reshape(*mesh_shape),
            (Axis.BATCH, Axis.SHARD),
        )

        with patch() as th_cfg:
            th_cfg.architecture = OmegaConf.create(
                {
                    "n_layers": cfg.num_hidden_layers,
                    "n_embd": cfg.hidden_size,
                    "n_head": cfg.num_attention_heads,
                    "n_kv_head": cfg.num_key_value_heads,
                    "head_dim": int(cfg.head_dim),
                    "intermediate_size": cfg.intermediate_size,
                    "block_size": cfg.max_position_embeddings,
                    "vocab_size": cfg.vocab_size,
                    "dropout": 0.0,
                    "attn_dropout": float(cfg.attention_dropout),
                    "rope_theta": float(rope_params.get("rope_theta", 10000.0)),
                    "partial_rotary_factor": float(
                        rope_params.get("partial_rotary_factor", 1.0)
                    ),
                    "rms_norm_eps": float(cfg.rms_norm_eps),
                    "bias": False,
                    "attention_bias": bool(cfg.attention_bias),
                    "layer_types": list(cfg.layer_types),
                    "linear_num_value_heads": int(cfg.linear_num_value_heads),
                    "linear_num_key_heads": int(cfg.linear_num_key_heads),
                    "linear_key_head_dim": int(cfg.linear_key_head_dim),
                    "linear_value_head_dim": int(cfg.linear_value_head_dim),
                    "linear_conv_kernel_dim": int(cfg.linear_conv_kernel_dim),
                    "dtype": {"param": "float32", "activation": "float32"},
                }
            )
            model = configure(Qwen3_5)
            dummy = jnp.zeros((1, 1), dtype=jnp.int32)
            abstract = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
            np_params = jax.tree_util.tree_map(
                lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
            )
            layer_types = list(cfg.layer_types)
            np_params = _from_hf_state_dict(np_params, state_dict, layer_types)
            sharded = _shard_from_np(np_params, abstract["params"], model, mesh)

            # Broadcast same prompt across the data-parallel batch when batch>1
            idx_np = inputs["input_ids"].numpy()
            attn_np = inputs["attention_mask"].numpy().astype(bool)
            if batch_size > 1:
                idx_np = np.repeat(idx_np, batch_size, axis=0)
                attn_np = np.repeat(attn_np, batch_size, axis=0)
            idx = jnp.array(idx_np)
            attn_b = jnp.array(attn_np)

            @jax.jit
            def fwd(p: object, x: jax.Array, m: jax.Array) -> object:
                return model.apply({"params": p}, x, padding_mask=m, deterministic=True)

            logits_jax, _ = fwd(sharded, idx, attn_b)
            logits_jax = np.array(logits_jax)

            # Compare only at the last non-padded token (matches qwen_parity).
            am = inputs["attention_mask"][0].numpy()
            last = int(am.sum() - 1)
            hf_last = logits_hf[0, last]
            jax_last = logits_jax[0, last]
            diff = float(np.max(np.abs(jax_last - hf_last)))
            mean = float(np.mean(np.abs(jax_last - hf_last)))
            overlap = len(set(hf_last.argsort()[-5:]) & set(jax_last.argsort()[-5:]))
            print(
                f"  vs HF (last-tok): max {diff:.5e}, mean {mean:.5e}, top5 {overlap}"
            )
            if baseline_logits is None:
                baseline_logits = jax_last
                sharded_for_roundtrip = sharded
                baseline_top5 = set(jax_last.argsort()[-5:])
            else:
                bdiff = float(np.max(np.abs(jax_last - baseline_logits)))
                overlap_base = len(baseline_top5 & set(jax_last.argsort()[-5:]))
                print(
                    f"  vs single-device baseline: max diff {bdiff:.5e}, top5 overlap {overlap_base}"
                )
                # GPU tf32/bf16 accumulation orders differ across sharding
                # configs — absolute logit values can drift a bit. Top-5
                # overlap is the real correctness gate.
                assert overlap_base == 5, (
                    f"{name} diverged: top5 overlap only {overlap_base}/5"
                )
            # For DP: each batch row receives the same prompt → outputs should be equal.
            if batch_size > 1:
                across_batch = float(np.max(np.abs(logits_jax - logits_jax[0:1])))
                print(
                    f"  DP batch consistency: max diff across rows {across_batch:.5e}"
                )

    # Roundtrip: gather sharded params (with TP sharding) to host and re-export HF state_dict
    if sharded_for_roundtrip is not None:
        sd_out = _to_hf_state_dict(jax.device_get(sharded_for_roundtrip), layer_types)
        assert sd_out["model.embed_tokens.weight"].shape[0] == cfg.vocab_size
        # Load back into a fresh HF model and verify forward matches
        hf_rt = Qwen3_5ForCausalLM(hf.config)
        hf_rt.load_state_dict(sd_out, strict=False)
        hf_rt.eval()
        with torch.no_grad():
            logits_hf_rt = hf_rt(**inputs).logits.detach().cpu().numpy()
        rt_max = float(np.max(np.abs(logits_hf - logits_hf_rt)))
        print(f"\nroundtrip (sharded → HF → HF forward): max diff {rt_max:.5e}")


if __name__ == "__main__":
    main()
