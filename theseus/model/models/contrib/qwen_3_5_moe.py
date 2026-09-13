"""Text-only Qwen 3.5 MoE (35B-A3B).

Subclasses :class:`Qwen3_5` — same hybrid attention layout, but each layer's
MLP is a routed top-k MoE with a sigmoid-gated shared expert.

HF stores experts as fused 3D tensors ``gate_up_proj`` / ``down_proj``. The
loader splits ``gate_up_proj`` along the intermediate axis into our separate
``gate`` / ``up`` vmapped MLPs.
"""

from typing import Any, List, Type

import flax.linen as nn
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from theseus.config import configure, field, patch
from theseus.model.block.qwen_3_5 import Qwen3_5MoEDecoderBlock
from theseus.model.models.contrib.qwen_3_5 import (
    Qwen3_5,
    _arch_from_hf,
    _assign,
    _load_full_attn,
    _load_gated_delta,
    _to_np,
)

try:
    import torch
except Exception:
    torch = None


class Qwen3_5MoE(Qwen3_5):
    n_layers: int = field("architecture/n_layers", default=40)
    n_embd: int = field("architecture/n_embd", default=2048)
    n_head: int = field("architecture/n_head", default=16)
    n_kv_head: int = field("architecture/n_kv_head", default=2)
    intermediate_size: int = field("architecture/intermediate_size", default=512)
    linear_num_value_heads: int = field(
        "architecture/linear_num_value_heads", default=32
    )
    # MoE
    num_experts: int = field("architecture/num_experts", default=256)
    num_experts_per_tok: int = field("architecture/num_experts_per_tok", default=8)
    moe_intermediate_size: int = field(
        "architecture/moe_intermediate_size", default=512
    )
    shared_expert_intermediate_size: int = field(
        "architecture/shared_expert_intermediate_size", default=512
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [Qwen3_5MoEDecoderBlock]

    def _make_block(self, layer_type: str) -> Any:
        return configure(Qwen3_5MoEDecoderBlock, layer_type=layer_type)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cpu",
        param_dtype: str = "float32",
        activation_dtype: str = "bfloat16",
    ) -> Any:
        import jax
        import jax.numpy as jnp
        import torch as _torch

        try:
            from transformers import Qwen3_5MoeForCausalLM
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformers", "tokenizers"}:
                raise
            raise ModuleNotFoundError(
                "Pretrained loading requires the huggingface dependency group. "
                "Install with `uv sync --group huggingface` "
                "or `pip install 'libthx[huggingface]'`."
            ) from exc

        # See qwen_3_5.Qwen3_5.from_pretrained — same dtype-honoring fix.
        # The MoE checkpoint is even larger (~35B) so the host-RAM and
        # single-chip materialization win matters more here.
        _TORCH_DTYPES = {
            "float32": _torch.float32,
            "fp32": _torch.float32,
            "bfloat16": _torch.bfloat16,
            "bf16": _torch.bfloat16,
            "float16": _torch.float16,
            "fp16": _torch.float16,
        }
        hf_model = Qwen3_5MoeForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_TORCH_DTYPES.get(param_dtype, _torch.float32),
            device_map=None,
        )
        hf_model.to(device)
        hf_model.eval()
        text_cfg = getattr(hf_model.config, "text_config", hf_model.config)
        arch = _arch_from_hf(text_cfg, param_dtype, activation_dtype)
        arch.num_experts = int(text_cfg.num_experts)
        arch.num_experts_per_tok = int(text_cfg.num_experts_per_tok)
        arch.moe_intermediate_size = int(text_cfg.moe_intermediate_size)
        arch.shared_expert_intermediate_size = int(
            text_cfg.shared_expert_intermediate_size
        )
        with patch() as th_cfg:
            if "architecture" in th_cfg:
                th_cfg.architecture = OmegaConf.merge(th_cfg.architecture, arch)
            else:
                th_cfg.architecture = arch
            model = configure(cls)
            dummy = jnp.zeros((1, 1), dtype=jnp.int32)
            abstract = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
            params = jax.tree_util.tree_map(
                lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
            )
        params = _from_hf_state_dict(
            params, hf_model.state_dict(), list(text_cfg.layer_types)
        )
        return model, params


def _load_moe(
    p: Any, block_key: str, state_dict: Any, prefix: str, moe_int: int
) -> None:
    """Load Qwen 3.5 MoE FFN: router + experts (fused gate_up split) + shared expert."""
    # Router: HF (E, H) -> our kernel (H, E)
    _assign(
        p,
        [block_key, "mlp", "router", "kernel"],
        _to_np(state_dict[prefix + "gate.weight"]).T,
    )

    # Experts: HF gate_up_proj is (E, 2I, H). Split into gate (E, I, H) and up
    # (E, I, H), then transpose each to (E, H, I) which matches our vmapped
    # QwenMLP gate/up.kernel layout.
    gate_up = _to_np(state_dict[prefix + "experts.gate_up_proj"])  # (E, 2I, H)
    gate_w = gate_up[:, :moe_int, :].transpose(0, 2, 1)  # (E, H, I)
    up_w = gate_up[:, moe_int:, :].transpose(0, 2, 1)
    _assign(p, [block_key, "mlp", "experts", "gate", "kernel"], gate_w)
    _assign(p, [block_key, "mlp", "experts", "up", "kernel"], up_w)
    # HF down_proj: (E, H, I) used as transpose, so our (E, I, H) = HF.transpose(0,2,1)
    down_hf = _to_np(state_dict[prefix + "experts.down_proj"])  # (E, H, I)
    _assign(
        p, [block_key, "mlp", "experts", "down", "kernel"], down_hf.transpose(0, 2, 1)
    )

    # Shared expert (regular QwenMLP)
    _assign(
        p,
        [block_key, "mlp", "shared_expert", "gate", "kernel"],
        _to_np(state_dict[prefix + "shared_expert.gate_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "mlp", "shared_expert", "up", "kernel"],
        _to_np(state_dict[prefix + "shared_expert.up_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "mlp", "shared_expert", "down", "kernel"],
        _to_np(state_dict[prefix + "shared_expert.down_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "mlp", "shared_expert_gate", "kernel"],
        _to_np(state_dict[prefix + "shared_expert_gate.weight"]).T,
    )


def _from_hf_state_dict(params: Any, state_dict: Any, layer_types: List[str]) -> Any:
    from flax.core import freeze, unfreeze

    p = unfreeze(params)
    n_layers = len(layer_types)

    logger.debug("loading embedding weights...")
    _assign(p, ["wte"], _to_np(state_dict["model.embed_tokens.weight"]))
    if "lm_head.weight" in state_dict:
        _assign(p, ["lm_head"], _to_np(state_dict["lm_head.weight"]))
    else:
        _assign(p, ["lm_head"], _to_np(state_dict["model.embed_tokens.weight"]))

    # Infer moe_intermediate_size from a sample experts.down_proj shape (E, H, I)
    sample_key = "model.layers.0.mlp.experts.down_proj"
    moe_int = state_dict[sample_key].shape[2]

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        block_key = f"blocks_{i}"

        _assign(
            p,
            [block_key, "rms_1", "weight"],
            _to_np(state_dict[prefix + "input_layernorm.weight"]),
        )
        _assign(
            p,
            [block_key, "rms_2", "weight"],
            _to_np(state_dict[prefix + "post_attention_layernorm.weight"]),
        )

        if layer_types[i] == "linear_attention":
            _load_gated_delta(p, block_key, state_dict, prefix + "linear_attn.")
        else:
            _load_full_attn(p, block_key, state_dict, prefix + "self_attn.")

        _load_moe(p, block_key, state_dict, prefix + "mlp.", moe_int)

    _assign(p, ["ln_f", "weight"], _to_np(state_dict["model.norm.weight"]))
    return freeze(p)


def _to_hf_state_dict(params: Any, layer_types: List[str]) -> Any:
    if torch is None:
        raise ImportError("torch is required for HF export")
    import jax
    from flax.core import unfreeze

    p = unfreeze(params)

    def grab(path: List[str]) -> np.ndarray:
        cur = p
        for key in path:
            cur = cur[key]
        if isinstance(cur, nn.Partitioned):
            cur = cur.value
        return np.array(jax.device_get(cur), dtype=np.float32)

    def t(x: np.ndarray) -> "torch.Tensor":
        return torch.tensor(x, dtype=torch.float32)

    sd: dict[str, "torch.Tensor"] = {}
    sd["model.embed_tokens.weight"] = t(grab(["wte"]))
    sd["lm_head.weight"] = t(grab(["lm_head"]))

    for i, lt in enumerate(layer_types):
        prefix = f"model.layers.{i}."
        block_key = f"blocks_{i}"

        sd[prefix + "input_layernorm.weight"] = t(grab([block_key, "rms_1", "weight"]))
        sd[prefix + "post_attention_layernorm.weight"] = t(
            grab([block_key, "rms_2", "weight"])
        )

        if lt == "linear_attention":
            for k_in, k_out in [
                ("in_proj_qkv", "in_proj_qkv"),
                ("in_proj_z", "in_proj_z"),
                ("in_proj_b", "in_proj_b"),
                ("in_proj_a", "in_proj_a"),
                ("out_proj", "out_proj"),
            ]:
                sd[prefix + f"linear_attn.{k_out}.weight"] = t(
                    grab([block_key, "attn", k_in, "kernel"]).T
                )
            sd[prefix + "linear_attn.conv1d.weight"] = t(
                grab([block_key, "attn", "conv_weight"])[:, None, :]
            )
            sd[prefix + "linear_attn.dt_bias"] = t(grab([block_key, "attn", "dt_bias"]))
            sd[prefix + "linear_attn.A_log"] = t(grab([block_key, "attn", "A_log"]))
            sd[prefix + "linear_attn.norm.weight"] = t(
                grab([block_key, "attn", "norm", "weight"])
            )
        else:
            for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sd[prefix + f"self_attn.{k}.weight"] = t(
                    grab([block_key, "attn", k, "kernel"]).T
                )
            sd[prefix + "self_attn.q_norm.weight"] = t(
                grab([block_key, "attn", "q_norm", "weight"])
            )
            sd[prefix + "self_attn.k_norm.weight"] = t(
                grab([block_key, "attn", "k_norm", "weight"])
            )

        # MoE export
        sd[prefix + "mlp.gate.weight"] = t(
            grab([block_key, "mlp", "router", "kernel"]).T
        )
        gate_w = grab([block_key, "mlp", "experts", "gate", "kernel"])  # (E, H, I)
        up_w = grab([block_key, "mlp", "experts", "up", "kernel"])
        down_w = grab([block_key, "mlp", "experts", "down", "kernel"])  # (E, I, H)
        # HF gate_up_proj: (E, 2I, H) = concat(gate.T, up.T)
        gate_t = gate_w.transpose(0, 2, 1)  # (E, I, H)
        up_t = up_w.transpose(0, 2, 1)
        sd[prefix + "mlp.experts.gate_up_proj"] = t(
            np.concatenate([gate_t, up_t], axis=1)
        )
        sd[prefix + "mlp.experts.down_proj"] = t(down_w.transpose(0, 2, 1))  # (E, H, I)

        for k_in, k_out in [
            ("gate", "gate_proj"),
            ("up", "up_proj"),
            ("down", "down_proj"),
        ]:
            sd[prefix + f"mlp.shared_expert.{k_out}.weight"] = t(
                grab([block_key, "mlp", "shared_expert", k_in, "kernel"]).T
            )
        sd[prefix + "mlp.shared_expert_gate.weight"] = t(
            grab([block_key, "mlp", "shared_expert_gate", "kernel"]).T
        )

    sd["model.norm.weight"] = t(grab(["ln_f", "weight"]))
    return sd


__all__ = ["Qwen3_5MoE", "_from_hf_state_dict", "_to_hf_state_dict"]
