"""HF<->JAX Qwen 3.5 MoE (text-only) parity check.

Usage: uv run python scripts/qwen_3.5_moe_parity.py --model Qwen/Qwen3.5-35B-A3B --prompt "Hello world"

Memory plan (35B-A3B):
  * HF loaded in bf16 on a single GPU (~70 GB). After the HF forward we
    materialize the state dict on CPU as fp32 (~140 GB) and free the GPU
    copy.
  * JAX params are sharded across the configured devices (default: all
    devices visible to JAX along the SHARD axis). Per-device cost is roughly
    ``35B * 4 / shard`` GB; 2 H100 NVL hosts the model comfortably.

The roundtrip step exports the JAX params back to HF format and only validates
shapes/key coverage — it does not reload a second HF model (which would not
fit alongside the JAX shards on the same devices).
"""

import argparse
import gc
import os

# Avoid XLA preallocating ~80% of GPU memory before we've placed sharded
# params; with a 35B model and 95 GB H100s every GB matters.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as flax_nn
import torch
from jax.sharding import Mesh
from omegaconf import OmegaConf

from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from transformers.utils import logging as hf_logging

from theseus.base.axis import Axis
from theseus.config import patch, configure
from theseus.model.models.contrib.qwen_3_5_moe import (
    Qwen3_5MoE,
    _from_hf_state_dict,
    _to_hf_state_dict,
)

hf_logging.set_verbosity_error()


def _shard_from_np(
    np_params: object, abstract: object, model: Qwen3_5MoE, mesh: Mesh
) -> object:
    """Convert a CPU PyTree (with flax Partitioned wrappers) to a sharded JAX
    PyTree using ``model.sharding`` rules and ``mesh``."""
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
    assert len(np_flat) == len(sharding_flat), (
        f"param/sharding leaf mismatch: {len(np_flat)} vs {len(sharding_flat)}"
    )
    sharded_flat = [
        jax.device_put(unwrap(a), s) for a, s in zip(np_flat, sharding_flat)
    ]
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(var_sharding), sharded_flat
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-length", type=int, default=64)
    # 35B-A3B fp32 sharded across 4 H100s ≈ 21 GB / device — fits with room
    # for activations and JIT and gives much tighter parity than bf16. Drop
    # to bf16 if you only have 2 visible devices.
    parser.add_argument("--param-dtype", default="float32")
    parser.add_argument("--activation-dtype", default="float32")
    args = parser.parse_args()

    # ---- HF forward on CPU ----
    # 35B fp32 on a single H100 (95 GB) doesn't fit, and JAX needs the full
    # GPU budget for sharded params. Run HF on CPU (fp32 — CPU MoE bf16 path
    # currently hits ``torch._grouped_mm`` alignment errors). Forward is slow
    # but parity scripts only do one prefill pass.
    hf = Qwen3_5MoeForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=None
    )
    hf.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    chat = [{"role": "user", "content": args.prompt}]
    prompt_text = tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    inputs_cpu = tok(
        prompt_text,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )

    print("running HF forward on CPU (slow for 35B; expect minutes)...")
    with torch.no_grad():
        logits_hf = hf(**inputs_cpu).logits.float().cpu().numpy()
        labels = inputs_cpu["input_ids"].clone()
        labels[inputs_cpu["attention_mask"] == 0] = -100
        loss_hf = hf(
            input_ids=inputs_cpu["input_ids"],
            attention_mask=inputs_cpu["attention_mask"],
            labels=labels,
        ).loss.item()

    cfg = getattr(hf.config, "text_config", hf.config)
    hf_full_cfg = hf.config
    state_dict_cpu = dict(hf.state_dict())

    del hf
    gc.collect()

    # ---- JAX setup ----
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", 10000.0)
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)

    devices = jax.devices()
    if len(devices) > 1:
        mesh = Mesh(
            np.array(devices).reshape(1, len(devices)), (Axis.BATCH, Axis.SHARD)
        )
    else:
        mesh = Mesh(np.array(devices).reshape(1, 1), (Axis.BATCH, Axis.SHARD))
    print(f"JAX devices: {devices}, mesh shape: {mesh.shape}")

    with patch() as th_cfg:
        th_cfg.architecture = OmegaConf.create(
            {
                "n_layers": cfg.num_hidden_layers,
                "n_embd": cfg.hidden_size,
                "n_head": cfg.num_attention_heads,
                "n_kv_head": cfg.num_key_value_heads,
                "head_dim": int(cfg.head_dim),
                "intermediate_size": int(cfg.moe_intermediate_size),
                "block_size": cfg.max_position_embeddings,
                "vocab_size": cfg.vocab_size,
                "dropout": 0.0,
                "attn_dropout": float(cfg.attention_dropout),
                "rope_theta": float(rope_theta),
                "partial_rotary_factor": float(partial_rotary_factor),
                "rms_norm_eps": float(cfg.rms_norm_eps),
                "bias": False,
                "attention_bias": bool(cfg.attention_bias),
                "layer_types": list(cfg.layer_types),
                "linear_num_value_heads": int(cfg.linear_num_value_heads),
                "linear_num_key_heads": int(cfg.linear_num_key_heads),
                "linear_key_head_dim": int(cfg.linear_key_head_dim),
                "linear_value_head_dim": int(cfg.linear_value_head_dim),
                "linear_conv_kernel_dim": int(cfg.linear_conv_kernel_dim),
                "num_experts": int(cfg.num_experts),
                "num_experts_per_tok": int(cfg.num_experts_per_tok),
                "moe_intermediate_size": int(cfg.moe_intermediate_size),
                "shared_expert_intermediate_size": int(
                    cfg.shared_expert_intermediate_size
                ),
                "dtype": {
                    "param": args.param_dtype,
                    "activation": args.activation_dtype,
                },
            }
        )

        model = configure(Qwen3_5MoE)
        dummy = jnp.zeros((1, 1), dtype=jnp.int32)
        abstract = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)

        # Build CPU np params via state dict
        params_np = jax.tree_util.tree_map(
            lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
        )
        layer_types = list(cfg.layer_types)
        params_np = _from_hf_state_dict(params_np, state_dict_cpu, layer_types)
        del state_dict_cpu
        gc.collect()

        # Shard onto the mesh.
        sharded_params = _shard_from_np(params_np, abstract["params"], model, mesh)
        del params_np
        gc.collect()

        idx = jnp.array(inputs_cpu["input_ids"].numpy())
        attn_bool = jnp.array(inputs_cpu["attention_mask"].numpy(), dtype=bool)

        @jax.jit
        def fwd(p: object, x: jax.Array, m: jax.Array) -> object:
            return model.apply({"params": p}, x, padding_mask=m, deterministic=True)

        logits_jax, _ = fwd(sharded_params, idx, attn_bool)
        logits_jax = np.array(logits_jax)

        attn = inputs_cpu["attention_mask"][0].numpy()
        last_tok_idx = int(attn.sum() - 1)
        logits_hf_last = logits_hf[0, last_tok_idx]
        logits_jax_last = logits_jax[0, last_tok_idx]
        max_diff = float(np.max(np.abs(logits_hf_last - logits_jax_last)))
        mean_diff = float(np.mean(np.abs(logits_hf_last - logits_jax_last)))
        overlap = len(
            set(logits_hf_last.argsort()[-5:]) & set(logits_jax_last.argsort()[-5:])
        )
        print(f"max diff: {max_diff}")
        print(f"mean diff: {mean_diff}")
        print(f"top5 overlap: {overlap}")

        # Cross-entropy loss on shifted tokens
        idx_np = inputs_cpu["input_ids"].numpy()
        attn_np = inputs_cpu["attention_mask"].numpy().astype(bool)
        logits_shift = logits_jax[:, :-1, :]
        targets = idx_np[:, 1:]
        mask = attn_np[:, 1:]
        log_probs = logits_shift - np.log(np.exp(logits_shift).sum(-1, keepdims=True))
        nll = -np.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        nll = nll * mask
        loss_jax = float(nll.sum() / mask.sum())
        print(f"hf loss: {loss_hf}")
        print(f"jax loss: {loss_jax}")

        # Round-trip back to HF state dict — verify shapes/keys without
        # reloading HF (would not fit alongside JAX shards on the same devices).
        sd = _to_hf_state_dict(jax.device_get(sharded_params), layer_types)
        assert sd["model.embed_tokens.weight"].shape[0] == cfg.vocab_size
        # Sanity: every layer's MoE has experts gate_up_proj + down_proj
        for i in range(cfg.num_hidden_layers):
            assert f"model.layers.{i}.mlp.experts.gate_up_proj" in sd
            assert f"model.layers.{i}.mlp.experts.down_proj" in sd
        print(f"export state_dict keys: {len(sd)}")
        del hf_full_cfg


if __name__ == "__main__":
    main()
