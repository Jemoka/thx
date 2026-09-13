from typing import Any, Optional, List, Tuple, Type

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn

from theseus.base.axis import ShardingPlan
from theseus.config import field, configure, patch
from omegaconf import OmegaConf
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.model.block.llama import LlamaDecoderBlock
from theseus.model.layers import RMSNorm

from loguru import logger

try:
    import torch  # optional, for HF weight export
except Exception:
    torch = None
from theseus.base.axis import Axis


class Marin(Module):
    """Marin model (marin-community/marin-8b-base).

    Architecturally identical to Llama 3 -- same decoder blocks, GQA, SiLU MLP,
    RMSNorm, and RoPE.  Defaults reflect the Marin-8B configuration.
    """

    n_layers: int = field("architecture/n_layers", default=32)
    n_embd: int = field("architecture/n_embd", default=4096)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=8)
    intermediate_size: int = field("architecture/intermediate_size", default=14336)
    rope_theta: float = field("architecture/rope_theta", default=500000.0)
    rms_norm_eps: float = field("architecture/rms_norm_eps", default=1e-5)
    block_size: int = field("architecture/block_size", default=4096)
    vocab_size: int = field("architecture/vocab_size", default=128256)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    bias: bool = field("architecture/bias", default=False)
    attention_bias: bool = field("architecture/attention_bias", default=False)

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
            ]
        )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LlamaDecoderBlock, RMSNorm]

    def flops(self, seq: int) -> float:
        return float(
            sum(block.flops(seq) for block in self.blocks)
            + 6 * seq * self.n_embd * self.vocab_size
        )

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        self.wte: Any = self.param(
            "wte",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )
        self.lm_head: Any = self.param(
            "lm_head",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )

        self.drop = nn.Dropout(rate=self.dropout)
        self.blocks = [configure(LlamaDecoderBlock) for _ in range(self.n_layers)]
        self.ln_f = configure(RMSNorm)

    def embed(self, idx: jax.Array, deterministic: bool = False) -> Any:
        x = jnp.take(self.wte, idx, axis=0).astype(self._activation_dtype)
        x = self.drop(x, deterministic=deterministic)
        return x

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
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
            )
        return x

    def unembed(self, x: jax.Array) -> Any:
        x = self.ln_f(x)
        x = x.astype(self._activation_dtype)
        head = jnp.asarray(self.lm_head, dtype=self._activation_dtype)
        logits = jnp.einsum("bth,vh->btv", x, head)
        return logits

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)
        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)
        loss = -jnp.sum(
            jnn.log_softmax(logits_flat, axis=-1)
            * jnn.one_hot(targets_masked, self.vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)
        return loss

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[Any, Optional[Any]]:
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        x = self.embed(idx, deterministic)
        x = self.decode(x, padding_mask=padding_mask, deterministic=deterministic)
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
        import torch

        try:
            from transformers import LlamaForCausalLM
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformers", "tokenizers"}:
                raise
            raise ModuleNotFoundError(
                "Pretrained loading requires the huggingface dependency group. "
                "Install with `uv sync --group huggingface` "
                "or `pip install 'libthx[huggingface]'`."
            ) from exc

        torch_dtype = torch.float32
        hf_model = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=None
        )
        hf_model.to(device)
        hf_model.eval()
        cfg = hf_model.config

        rope_theta = getattr(cfg, "rope_theta", 500000.0)

        with patch() as th_cfg:
            new_arch = OmegaConf.create(
                {
                    "n_layers": cfg.num_hidden_layers,
                    "n_embd": cfg.hidden_size,
                    "n_head": cfg.num_attention_heads,
                    "n_kv_head": cfg.num_key_value_heads,
                    "intermediate_size": cfg.intermediate_size,
                    "block_size": cfg.max_position_embeddings,
                    "vocab_size": cfg.vocab_size,
                    "dropout": 0.0,
                    "attn_dropout": float(cfg.attention_dropout),
                    "rope_theta": float(rope_theta),
                    "rms_norm_eps": float(cfg.rms_norm_eps),
                    "bias": cfg.mlp_bias if hasattr(cfg, "mlp_bias") else False,
                    "attention_bias": cfg.attention_bias
                    if hasattr(cfg, "attention_bias")
                    else False,
                    "partial_rotary_factor": 1.0,
                    "use_sliding_window": False,
                    "sliding_window": -1,
                    "dtype": {"param": param_dtype, "activation": activation_dtype},
                }
            )
            if "architecture" in th_cfg:
                th_cfg.architecture = OmegaConf.merge(th_cfg.architecture, new_arch)
            else:
                th_cfg.architecture = new_arch

            model = configure(cls)
            dummy = jnp.zeros((1, 1), dtype=jnp.int32)
            abstract = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
            params = jax.tree_util.tree_map(
                lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
            )
        params = _from_hf_state_dict(params, hf_model.state_dict(), model.n_layers, cfg)
        return model, params


def _from_hf_state_dict(params: Any, state_dict: Any, n_layers: int, cfg: Any) -> Any:
    from flax.core import freeze, unfreeze

    p = unfreeze(params)

    def assign(path: list[str], array: Any) -> None:
        cur = p
        for key in path[:-1]:
            cur = cur[key]
        existing = cur[path[-1]]
        if isinstance(existing, nn.Partitioned):
            cur[path[-1]] = existing.replace(value=array)
        else:
            cur[path[-1]] = array

    # Embeddings and head
    logger.debug("loading embedding weights...")
    assign(["wte"], state_dict["model.embed_tokens.weight"].cpu().float().numpy())
    if "lm_head.weight" in state_dict:
        assign(["lm_head"], state_dict["lm_head.weight"].cpu().float().numpy())
    else:
        assign(
            ["lm_head"], state_dict["model.embed_tokens.weight"].cpu().float().numpy()
        )

    attn_has_bias = getattr(cfg, "attention_bias", False)
    mlp_has_bias = getattr(cfg, "mlp_bias", False)

    for i in range(n_layers):
        logger.debug(f"loading block {i} weights...")
        prefix = f"model.layers.{i}."
        block_key = f"blocks_{i}"
        # Norms
        assign(
            [block_key, "rms_1", "weight"],
            state_dict[prefix + "input_layernorm.weight"].cpu().float().numpy(),
        )
        assign(
            [block_key, "rms_2", "weight"],
            state_dict[prefix + "post_attention_layernorm.weight"]
            .cpu()
            .float()
            .numpy(),
        )

        # Attention projections
        q_w = state_dict[prefix + "self_attn.q_proj.weight"].cpu().float().numpy().T
        k_w = state_dict[prefix + "self_attn.k_proj.weight"].cpu().float().numpy().T
        v_w = state_dict[prefix + "self_attn.v_proj.weight"].cpu().float().numpy().T
        o_w = state_dict[prefix + "self_attn.o_proj.weight"].cpu().float().numpy().T

        assign([block_key, "attn", "q_proj", "kernel"], q_w)
        assign([block_key, "attn", "k_proj", "kernel"], k_w)
        assign([block_key, "attn", "v_proj", "kernel"], v_w)
        assign([block_key, "attn", "o_proj", "kernel"], o_w)

        if attn_has_bias:
            assign(
                [block_key, "attn", "q_proj", "bias"],
                state_dict[prefix + "self_attn.q_proj.bias"].cpu().float().numpy(),
            )
            assign(
                [block_key, "attn", "k_proj", "bias"],
                state_dict[prefix + "self_attn.k_proj.bias"].cpu().float().numpy(),
            )
            assign(
                [block_key, "attn", "v_proj", "bias"],
                state_dict[prefix + "self_attn.v_proj.bias"].cpu().float().numpy(),
            )

        # MLP
        gate_w = state_dict[prefix + "mlp.gate_proj.weight"].cpu().float().numpy().T
        up_w = state_dict[prefix + "mlp.up_proj.weight"].cpu().float().numpy().T
        down_w = state_dict[prefix + "mlp.down_proj.weight"].cpu().float().numpy().T
        assign([block_key, "mlp", "gate", "kernel"], gate_w)
        assign([block_key, "mlp", "up", "kernel"], up_w)
        assign([block_key, "mlp", "down", "kernel"], down_w)

        if mlp_has_bias:
            assign(
                [block_key, "mlp", "gate", "bias"],
                state_dict[prefix + "mlp.gate_proj.bias"].cpu().float().numpy(),
            )
            assign(
                [block_key, "mlp", "up", "bias"],
                state_dict[prefix + "mlp.up_proj.bias"].cpu().float().numpy(),
            )
            assign(
                [block_key, "mlp", "down", "bias"],
                state_dict[prefix + "mlp.down_proj.bias"].cpu().float().numpy(),
            )

    # Final norm
    assign(["ln_f", "weight"], state_dict["model.norm.weight"].cpu().float().numpy())

    logger.debug("model loading complete.")
    return freeze(p)


def _to_hf_state_dict(
    params: Any, n_layers: int, cfg: Any
) -> dict[str, "torch.Tensor"]:
    if torch is None:
        raise ImportError("torch is required for HF export")

    from flax.core import unfreeze

    p = unfreeze(params)
    sd: dict[str, torch.Tensor] = {}

    def grab(path: list[str]) -> Any:
        cur = p
        for key in path:
            cur = cur[key]
        if isinstance(cur, nn.Partitioned):
            cur = cur.value
        return np.array(jax.device_get(cur), dtype=np.float32)

    attn_has_bias = getattr(cfg, "attention_bias", False)
    mlp_has_bias = getattr(cfg, "mlp_bias", False)

    # Embeddings
    embed = torch.tensor(grab(["wte"]), dtype=torch.float32)
    head = torch.tensor(grab(["lm_head"]), dtype=torch.float32)
    sd["model.embed_tokens.weight"] = embed
    sd["lm_head.weight"] = head

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        block_key = f"blocks_{i}"

        sd[prefix + "input_layernorm.weight"] = torch.tensor(
            grab([block_key, "rms_1", "weight"]), dtype=torch.float32
        )
        sd[prefix + "post_attention_layernorm.weight"] = torch.tensor(
            grab([block_key, "rms_2", "weight"]), dtype=torch.float32
        )

        q_w = grab([block_key, "attn", "q_proj", "kernel"]).T
        k_w = grab([block_key, "attn", "k_proj", "kernel"]).T
        v_w = grab([block_key, "attn", "v_proj", "kernel"]).T
        o_w = grab([block_key, "attn", "o_proj", "kernel"]).T
        sd[prefix + "self_attn.q_proj.weight"] = torch.tensor(q_w, dtype=torch.float32)
        sd[prefix + "self_attn.k_proj.weight"] = torch.tensor(k_w, dtype=torch.float32)
        sd[prefix + "self_attn.v_proj.weight"] = torch.tensor(v_w, dtype=torch.float32)
        sd[prefix + "self_attn.o_proj.weight"] = torch.tensor(o_w, dtype=torch.float32)

        if attn_has_bias:
            sd[prefix + "self_attn.q_proj.bias"] = torch.tensor(
                grab([block_key, "attn", "q_proj", "bias"]), dtype=torch.float32
            )
            sd[prefix + "self_attn.k_proj.bias"] = torch.tensor(
                grab([block_key, "attn", "k_proj", "bias"]), dtype=torch.float32
            )
            sd[prefix + "self_attn.v_proj.bias"] = torch.tensor(
                grab([block_key, "attn", "v_proj", "bias"]), dtype=torch.float32
            )

        gate_w = grab([block_key, "mlp", "gate", "kernel"]).T
        up_w = grab([block_key, "mlp", "up", "kernel"]).T
        down_w = grab([block_key, "mlp", "down", "kernel"]).T
        sd[prefix + "mlp.gate_proj.weight"] = torch.tensor(gate_w, dtype=torch.float32)
        sd[prefix + "mlp.up_proj.weight"] = torch.tensor(up_w, dtype=torch.float32)
        sd[prefix + "mlp.down_proj.weight"] = torch.tensor(down_w, dtype=torch.float32)

        if mlp_has_bias:
            sd[prefix + "mlp.gate_proj.bias"] = torch.tensor(
                grab([block_key, "mlp", "gate", "bias"]), dtype=torch.float32
            )
            sd[prefix + "mlp.up_proj.bias"] = torch.tensor(
                grab([block_key, "mlp", "up", "bias"]), dtype=torch.float32
            )
            sd[prefix + "mlp.down_proj.bias"] = torch.tensor(
                grab([block_key, "mlp", "down", "bias"]), dtype=torch.float32
            )

    sd["model.norm.weight"] = torch.tensor(
        grab(["ln_f", "weight"]), dtype=torch.float32
    )
    return sd


__all__ = ["Marin", "_from_hf_state_dict", "_to_hf_state_dict"]
