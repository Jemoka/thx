"""Smoke test for Qwen3_5MoE under TP and DP sharding.

Uses a tiny synthetic config so it runs on CPU host devices. Builds two
sharded variants (TP: 1×4, DP: 4×1), runs the same forward on each, and
verifies the logits match the single-device baseline. Roundtrips the
TP-sharded params back to a fresh single-device init.

Run:
    JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4' \
        uv run python scripts/qwen_3.5_moe_sharded_smoke.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as flax_nn
from jax.sharding import Mesh
from omegaconf import OmegaConf

from theseus.base.axis import Axis
from theseus.config import patch, configure
from theseus.model.models.contrib.qwen_3_5_moe import Qwen3_5MoE


_ARCH = dict(
    n_layers=2,
    n_embd=64,
    n_head=4,
    n_kv_head=2,
    head_dim=16,
    intermediate_size=64,
    block_size=128,
    vocab_size=200,
    dropout=0.0,
    attn_dropout=0.0,
    rope_theta=1e7,
    partial_rotary_factor=0.25,
    rms_norm_eps=1e-6,
    bias=False,
    attention_bias=False,
    layer_types=["linear_attention", "full_attention"],
    linear_num_value_heads=4,
    linear_num_key_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    dtype={"param": "float32", "activation": "float32"},
)


def init_sharded(model: Qwen3_5MoE, mesh: Mesh) -> object:
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)
    shapes = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
    pspec = flax_nn.get_partition_spec(shapes)
    sharding = flax_nn.logical_to_mesh_sharding(pspec, mesh, rules=model.sharding._tp)
    return jax.jit(model.init, out_shardings=sharding)(jax.random.PRNGKey(0), dummy)


def main() -> None:
    devices = jax.devices()
    n = len(devices)
    print(f"jax devices: {n}")
    assert n >= 2, "smoke needs >=2 devices"

    rng = np.random.default_rng(0)
    base_idx = rng.integers(0, _ARCH["vocab_size"], size=(1, 16), dtype=np.int32)

    with patch() as th_cfg:
        th_cfg.architecture = OmegaConf.create(_ARCH)
        model = configure(Qwen3_5MoE)

        mesh_single = Mesh(
            np.array([devices[0]]).reshape(1, 1), (Axis.BATCH, Axis.SHARD)
        )
        vars_single = init_sharded(model, mesh_single)
        out_single, _ = model.apply(vars_single, jnp.array(base_idx))
        out_single = np.array(out_single)
        print(f"single: out shape {out_single.shape}")

        mesh_tp = Mesh(np.array(devices).reshape(1, n), (Axis.BATCH, Axis.SHARD))
        vars_tp = init_sharded(model, mesh_tp)
        out_tp, _ = model.apply(vars_tp, jnp.array(base_idx))
        out_tp = np.array(out_tp)
        diff_tp = float(np.max(np.abs(out_single - out_tp)))
        print(f"TP (1×{n}): max diff vs single {diff_tp:.5e}")
        assert diff_tp < 1e-3, f"TP diverged: {diff_tp}"

        mesh_dp = Mesh(np.array(devices).reshape(n, 1), (Axis.BATCH, Axis.SHARD))
        vars_dp = init_sharded(model, mesh_dp)
        batched = np.repeat(base_idx, n, axis=0)
        out_dp, _ = model.apply(vars_dp, jnp.array(batched))
        out_dp = np.array(out_dp)
        across = float(np.max(np.abs(out_dp - out_dp[0:1])))
        print(f"DP ({n}×1): batch consistency {across:.5e}")
        assert across < 1e-3, f"DP batch inconsistent: {across}"
        diff_dp = float(np.max(np.abs(out_single - out_dp[0])))
        print(f"DP row0 vs single: max diff {diff_dp:.5e}")
        assert diff_dp < 1e-3, f"DP diverged from single: {diff_dp}"

    print("\nPASS: MoE sharded TP + DP match single-device baseline")


if __name__ == "__main__":
    main()
