"""Text-only Qwen 3.5 with hybrid (full + linear) attention layers.

Vision keys in the HF checkpoint are skipped: loader is tolerant via
``strict=False`` on roundtrip.
"""

from typing import Any, List, Optional, Tuple, Type

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from theseus.base.axis import Axis
from theseus.base.axis import ShardingPlan
from theseus.config import configure, field, patch
from theseus.model.axes import Axes
from theseus.model.block.qwen_3_5 import Qwen3_5DecoderBlock
from theseus.model.layers import RMSNorm
from theseus.model.module import Module

try:
    import torch
except Exception:
    torch = None


class Qwen3_5(Module):
    n_layers: int = field("architecture/n_layers", default=24)
    n_embd: int = field("architecture/n_embd", default=1024)
    n_head: int = field("architecture/n_head", default=8)
    n_kv_head: int = field("architecture/n_kv_head", default=2)
    head_dim: int = field("architecture/head_dim", default=256)
    intermediate_size: int = field("architecture/intermediate_size", default=3584)
    rope_theta: float = field("architecture/rope_theta", default=10000000.0)
    partial_rotary_factor: float = field(
        "architecture/partial_rotary_factor", default=0.25
    )
    rms_norm_eps: float = field("architecture/rms_norm_eps", default=1e-6)
    block_size: int = field("architecture/block_size", default=262144)
    vocab_size: int = field("architecture/vocab_size", default=248320)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    bias: bool = field("architecture/bias", default=False)
    attention_bias: bool = field("architecture/attention_bias", default=False)
    layer_types: Any = field("architecture/layer_types", default_factory=list)
    # gated delta-net dims
    linear_num_value_heads: int = field(
        "architecture/linear_num_value_heads", default=16
    )
    linear_num_key_heads: int = field("architecture/linear_num_key_heads", default=16)
    linear_key_head_dim: int = field("architecture/linear_key_head_dim", default=128)
    linear_value_head_dim: int = field(
        "architecture/linear_value_head_dim", default=128
    )
    linear_conv_kernel_dim: int = field(
        "architecture/linear_conv_kernel_dim", default=4
    )

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(
            tp=[
                (Axes.VOCAB.value, None),
                (Axes.BLOCK_SIZE.value, None),
                (Axes.N_EMBD.value, None),
                (Axes.N_EMBD_FF.value, Axis.SHARD),
                (Axes.N_EMBD_OUT.value, Axis.SHARD),
                (Axes.N_ATTN.value, Axis.SHARD),
                (Axes.N_EXPERT.value, Axis.SHARD),
            ]
        )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [Qwen3_5DecoderBlock, RMSNorm]

    def _make_block(self, layer_type: str) -> Any:
        return configure(Qwen3_5DecoderBlock, layer_type=layer_type)

    def flops(self, seq: int) -> float:
        return float(
            sum(block.flops(seq) for block in self.blocks)
            + 6 * seq * self.n_embd * self.vocab_size
        )

    def setup(self) -> None:
        self.wte = self.param(
            "wte",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )
        self.lm_head = self.param(
            "lm_head",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )
        self.drop = nn.Dropout(rate=self.dropout)
        types = (
            list(self.layer_types)
            if self.layer_types
            else ["full_attention"] * self.n_layers
        )
        assert len(types) == self.n_layers, (
            f"layer_types has {len(types)} entries but n_layers={self.n_layers}"
        )
        self.blocks = [self._make_block(lt) for lt in types]
        self.ln_f = configure(RMSNorm, centered=True)

    def embed(self, idx: jax.Array, deterministic: bool = False) -> Any:
        wte = jnp.asarray(self.wte)
        x = jnp.take(wte, idx, axis=0).astype(self._activation_dtype)
        return self.drop(x, deterministic=deterministic)

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        cache_max_len: Optional[int] = None,
    ) -> jax.Array:
        b, t, _ = x.shape
        if padding_mask is None:
            positions = jnp.arange(t)
        else:
            positions = jnp.maximum(jnp.cumsum(padding_mask, axis=-1) - 1, 0)
        for block in self.blocks:
            x = block(
                x,
                padding_mask=padding_mask,
                deterministic=deterministic,
                positions=positions,
                cache_max_len=cache_max_len,
            )
        return x

    def unembed(self, x: jax.Array) -> Any:
        x = self.ln_f(x)
        x = x.astype(self._activation_dtype)
        head = jnp.asarray(self.lm_head, dtype=self._activation_dtype)
        return jnp.einsum("bth,vh->btv", x, head)

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)
        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)
        return -jnp.sum(
            jnn.log_softmax(logits_flat, axis=-1)
            * jnn.one_hot(targets_masked, self.vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        cache_max_len: Optional[int] = None,
    ) -> Tuple[Any, Optional[Any]]:
        x = self.embed(idx, deterministic)
        x = self.decode(
            x,
            padding_mask=padding_mask,
            deterministic=deterministic,
            cache_max_len=cache_max_len,
        )
        logits = self.unembed(x)
        loss_val = self.loss(logits, targets) if targets is not None else None
        return logits, loss_val

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cpu",
        param_dtype: str = "float32",
        activation_dtype: str = "bfloat16",
    ) -> Any:
        import torch as _torch

        try:
            from transformers import Qwen3_5ForCausalLM
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformers", "tokenizers"}:
                raise
            raise ModuleNotFoundError(
                "Pretrained loading requires the huggingface dependency group. "
                "Install with `uv sync --group huggingface` "
                "or `pip install 'libthx[huggingface]'`."
            ) from exc

        # Honor the requested param_dtype for the torch load itself, not
        # just for our jax-side abstract pytree.  Loading a 27B/35B
        # checkpoint as fp32 (~108/140 GB host RAM) when downstream uses
        # bf16 wastes ~half the host memory and pushes single-chip
        # materialization peaks past B200's 192 GB during the subsequent
        # jit-with-out_shardings reshard.
        _TORCH_DTYPES = {
            "float32": _torch.float32,
            "fp32": _torch.float32,
            "bfloat16": _torch.bfloat16,
            "bf16": _torch.bfloat16,
            "float16": _torch.float16,
            "fp16": _torch.float16,
        }
        hf_model = Qwen3_5ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_TORCH_DTYPES.get(param_dtype, _torch.float32),
            device_map=None,
        )
        hf_model.to(device)
        hf_model.eval()
        text_cfg = getattr(hf_model.config, "text_config", hf_model.config)
        arch = _arch_from_hf(text_cfg, param_dtype, activation_dtype)
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


def _arch_from_hf(cfg: Any, param_dtype: str, activation_dtype: str) -> Any:
    rope = getattr(cfg, "rope_parameters", None) or {}
    arch: dict[str, Any] = {
        "n_layers": cfg.num_hidden_layers,
        "n_embd": cfg.hidden_size,
        "n_head": cfg.num_attention_heads,
        "n_kv_head": cfg.num_key_value_heads,
        "head_dim": int(cfg.head_dim),
        "block_size": cfg.max_position_embeddings,
        "vocab_size": cfg.vocab_size,
        "dropout": 0.0,
        "attn_dropout": float(cfg.attention_dropout),
        "rope_theta": float(rope.get("rope_theta", 10000.0)),
        "partial_rotary_factor": float(rope.get("partial_rotary_factor", 1.0)),
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "bias": False,
        "attention_bias": bool(cfg.attention_bias),
        "layer_types": list(cfg.layer_types),
        "linear_num_value_heads": int(cfg.linear_num_value_heads),
        "linear_num_key_heads": int(cfg.linear_num_key_heads),
        "linear_key_head_dim": int(cfg.linear_key_head_dim),
        "linear_value_head_dim": int(cfg.linear_value_head_dim),
        "linear_conv_kernel_dim": int(cfg.linear_conv_kernel_dim),
        "dtype": {"param": param_dtype, "activation": activation_dtype},
    }
    if hasattr(cfg, "intermediate_size"):
        arch["intermediate_size"] = cfg.intermediate_size
    return OmegaConf.create(arch)


def _to_np(t: Any) -> np.ndarray:
    arr: np.ndarray = t.detach().cpu().float().numpy()
    return arr


def _assign(params: Any, path: List[str], array: np.ndarray) -> None:
    cur = params
    for key in path[:-1]:
        cur = cur[key]
    existing = cur[path[-1]]
    if isinstance(existing, nn.Partitioned):
        cast = array.astype(existing.value.dtype, copy=False)
        cur[path[-1]] = existing.replace(value=cast)
    else:
        cur[path[-1]] = array.astype(existing.dtype, copy=False)


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

        _load_mlp(p, block_key, state_dict, prefix + "mlp.")

    _assign(p, ["ln_f", "weight"], _to_np(state_dict["model.norm.weight"]))
    return freeze(p)


def _load_full_attn(p: Any, block_key: str, state_dict: Any, prefix: str) -> None:
    """Full-attention layers: GQA + Q-output-gate + Q/K head-norm."""
    _assign(
        p,
        [block_key, "attn", "q_proj", "kernel"],
        _to_np(state_dict[prefix + "q_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "k_proj", "kernel"],
        _to_np(state_dict[prefix + "k_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "v_proj", "kernel"],
        _to_np(state_dict[prefix + "v_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "o_proj", "kernel"],
        _to_np(state_dict[prefix + "o_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "q_norm", "weight"],
        _to_np(state_dict[prefix + "q_norm.weight"]),
    )
    _assign(
        p,
        [block_key, "attn", "k_norm", "weight"],
        _to_np(state_dict[prefix + "k_norm.weight"]),
    )


def _load_gated_delta(p: Any, block_key: str, state_dict: Any, prefix: str) -> None:
    """Linear-attention layers: gated delta-net + causal conv1d."""
    _assign(
        p,
        [block_key, "attn", "in_proj_qkv", "kernel"],
        _to_np(state_dict[prefix + "in_proj_qkv.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "in_proj_z", "kernel"],
        _to_np(state_dict[prefix + "in_proj_z.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "in_proj_b", "kernel"],
        _to_np(state_dict[prefix + "in_proj_b.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "in_proj_a", "kernel"],
        _to_np(state_dict[prefix + "in_proj_a.weight"]).T,
    )
    _assign(
        p,
        [block_key, "attn", "out_proj", "kernel"],
        _to_np(state_dict[prefix + "out_proj.weight"]).T,
    )
    # conv1d weight: (conv_dim, 1, K) -> (conv_dim, K)
    conv_w = _to_np(state_dict[prefix + "conv1d.weight"]).squeeze(1)
    _assign(p, [block_key, "attn", "conv_weight"], conv_w)
    _assign(p, [block_key, "attn", "dt_bias"], _to_np(state_dict[prefix + "dt_bias"]))
    _assign(p, [block_key, "attn", "A_log"], _to_np(state_dict[prefix + "A_log"]))
    _assign(
        p,
        [block_key, "attn", "norm", "weight"],
        _to_np(state_dict[prefix + "norm.weight"]),
    )


def _load_mlp(p: Any, block_key: str, state_dict: Any, prefix: str) -> None:
    _assign(
        p,
        [block_key, "mlp", "gate", "kernel"],
        _to_np(state_dict[prefix + "gate_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "mlp", "up", "kernel"],
        _to_np(state_dict[prefix + "up_proj.weight"]).T,
    )
    _assign(
        p,
        [block_key, "mlp", "down", "kernel"],
        _to_np(state_dict[prefix + "down_proj.weight"]).T,
    )


def _to_hf_state_dict(params: Any, layer_types: List[str]) -> Any:
    """Round-trip back to HF format. Vision keys are not produced — caller
    should ``load_state_dict(..., strict=False)`` on a full multimodal config.
    """
    if torch is None:
        raise ImportError("torch is required for HF export")
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
            sd[prefix + "linear_attn.in_proj_qkv.weight"] = t(
                grab([block_key, "attn", "in_proj_qkv", "kernel"]).T
            )
            sd[prefix + "linear_attn.in_proj_z.weight"] = t(
                grab([block_key, "attn", "in_proj_z", "kernel"]).T
            )
            sd[prefix + "linear_attn.in_proj_b.weight"] = t(
                grab([block_key, "attn", "in_proj_b", "kernel"]).T
            )
            sd[prefix + "linear_attn.in_proj_a.weight"] = t(
                grab([block_key, "attn", "in_proj_a", "kernel"]).T
            )
            sd[prefix + "linear_attn.out_proj.weight"] = t(
                grab([block_key, "attn", "out_proj", "kernel"]).T
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
            sd[prefix + "self_attn.q_proj.weight"] = t(
                grab([block_key, "attn", "q_proj", "kernel"]).T
            )
            sd[prefix + "self_attn.k_proj.weight"] = t(
                grab([block_key, "attn", "k_proj", "kernel"]).T
            )
            sd[prefix + "self_attn.v_proj.weight"] = t(
                grab([block_key, "attn", "v_proj", "kernel"]).T
            )
            sd[prefix + "self_attn.o_proj.weight"] = t(
                grab([block_key, "attn", "o_proj", "kernel"]).T
            )
            sd[prefix + "self_attn.q_norm.weight"] = t(
                grab([block_key, "attn", "q_norm", "weight"])
            )
            sd[prefix + "self_attn.k_norm.weight"] = t(
                grab([block_key, "attn", "k_norm", "weight"])
            )

        sd[prefix + "mlp.gate_proj.weight"] = t(
            grab([block_key, "mlp", "gate", "kernel"]).T
        )
        sd[prefix + "mlp.up_proj.weight"] = t(
            grab([block_key, "mlp", "up", "kernel"]).T
        )
        sd[prefix + "mlp.down_proj.weight"] = t(
            grab([block_key, "mlp", "down", "kernel"]).T
        )

    sd["model.norm.weight"] = t(grab(["ln_f", "weight"]))
    return sd


__all__ = ["Qwen3_5", "_from_hf_state_dict", "_to_hf_state_dict"]
