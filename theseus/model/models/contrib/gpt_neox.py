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
from theseus.model.block.gpt_neox import GPTNeoXDecoderBlock
from theseus.model.layers import LayerNorm

from loguru import logger

try:
    import torch  # optional, for HF weight export
except Exception:
    torch = None
from theseus.base.axis import Axis


class GPTNeoX(Module):
    n_layers: int = field("architecture/n_layers", default=24)
    n_embd: int = field("architecture/n_embd", default=2048)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=-1)
    intermediate_size: int = field("architecture/intermediate_size", default=8192)
    rope_theta: float = field("architecture/rope_theta", default=10000.0)
    partial_rotary_factor: float = field(
        "architecture/partial_rotary_factor", default=1.0
    )
    layer_norm_eps: float = field("architecture/layer_norm_eps", default=1e-5)
    block_size: int = field("architecture/block_size", default=2048)
    vocab_size: int = field("architecture/vocab_size", default=50432)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    bias: bool = field("architecture/bias", default=True)
    attention_bias: bool = field("architecture/attention_bias", default=True)
    use_parallel_residual: bool = field(
        "architecture/use_parallel_residual", default=True
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
            ]
        )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [GPTNeoXDecoderBlock, LayerNorm]

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
        self.blocks = [configure(GPTNeoXDecoderBlock) for _ in range(self.n_layers)]
        self.ln_f = configure(LayerNorm)

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
            from transformers import GPTNeoXForCausalLM
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformers", "tokenizers"}:
                raise
            raise ModuleNotFoundError(
                "Pretrained loading requires the huggingface dependency group. "
                "Install with `uv sync --group huggingface` "
                "or `pip install 'libthx[huggingface]'`."
            ) from exc

        torch_dtype = torch.float32
        hf_model = GPTNeoXForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=None
        )
        hf_model.to(device)
        hf_model.eval()
        cfg = hf_model.config

        rope_theta = getattr(cfg, "rotary_emb_base", 10000.0)
        partial_factor = getattr(cfg, "rotary_pct", 1.0)
        if cfg.rope_parameters is not None:
            rope_theta = cfg.rope_parameters.get("rope_theta", rope_theta)
            partial_factor = cfg.rope_parameters.get(
                "partial_rotary_factor", partial_factor
            )

        n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        with patch() as th_cfg:
            new_arch = OmegaConf.create(
                {
                    "n_layers": cfg.num_hidden_layers,
                    "n_embd": cfg.hidden_size,
                    "n_head": cfg.num_attention_heads,
                    "n_kv_head": n_kv,
                    "intermediate_size": cfg.intermediate_size,
                    "block_size": cfg.max_position_embeddings,
                    "vocab_size": cfg.vocab_size,
                    "dropout": float(cfg.hidden_dropout),
                    "attn_dropout": float(cfg.attention_dropout),
                    "rope_theta": float(rope_theta),
                    "partial_rotary_factor": float(partial_factor),
                    "layer_norm_eps": float(cfg.layer_norm_eps),
                    "bias": cfg.mlp_bias if hasattr(cfg, "mlp_bias") else True,
                    "attention_bias": cfg.attention_bias
                    if hasattr(cfg, "attention_bias")
                    else True,
                    "use_parallel_residual": bool(cfg.use_parallel_residual),
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

    # Embeddings (tied)
    logger.debug("loading embedding weights...")
    embed = state_dict["gpt_neox.embed_in.weight"].cpu().float().numpy()
    assign(["wte"], embed)
    if "embed_out.weight" in state_dict:
        assign(["lm_head"], state_dict["embed_out.weight"].cpu().float().numpy())
    else:
        assign(["lm_head"], embed)

    for i in range(n_layers):
        logger.debug(f"loading block {i} weights...")
        prefix = f"gpt_neox.layers.{i}."
        block_key = f"blocks_{i}"
        # Norms
        logger.debug(f"  loading block {i} norm weights...")
        assign(
            [block_key, "ln_1", "weight"],
            state_dict[prefix + "input_layernorm.weight"].cpu().float().numpy(),
        )
        assign(
            [block_key, "ln_1", "bias"],
            state_dict[prefix + "input_layernorm.bias"].cpu().float().numpy(),
        )
        logger.debug(f"  loading block {i} post-attention norm weights...")
        assign(
            [block_key, "ln_2", "weight"],
            state_dict[prefix + "post_attention_layernorm.weight"]
            .cpu()
            .float()
            .numpy(),
        )
        assign(
            [block_key, "ln_2", "bias"],
            state_dict[prefix + "post_attention_layernorm.bias"].cpu().float().numpy(),
        )

        # Attention projections (qkv fused, interleaved per head)
        # HF layout: (3*hidden, hidden) with output dim interleaved as
        # [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...] per head
        num_heads = cfg.num_attention_heads
        hidden = cfg.hidden_size
        head_dim = hidden // num_heads
        qkv = (
            state_dict[prefix + "attention.query_key_value.weight"]
            .cpu()
            .float()
            .numpy()
        )
        qkv_b = (
            state_dict[prefix + "attention.query_key_value.bias"].cpu().float().numpy()
        )
        # De-interleave: reshape output dim to (num_heads, 3, head_dim)
        qkv_r = qkv.reshape(num_heads, 3, head_dim, hidden)
        q_w = qkv_r[:, 0, :, :].reshape(hidden, hidden).T  # (in, out) for JAX
        k_w = qkv_r[:, 1, :, :].reshape(hidden, hidden).T
        v_w = qkv_r[:, 2, :, :].reshape(hidden, hidden).T
        logger.debug(f"  loading block {i} attention weights...")
        assign([block_key, "attn", "q_proj", "kernel"], q_w)
        assign([block_key, "attn", "k_proj", "kernel"], k_w)
        assign([block_key, "attn", "v_proj", "kernel"], v_w)

        qkv_b_r = qkv_b.reshape(num_heads, 3, head_dim)
        q_b = qkv_b_r[:, 0, :].reshape(hidden)
        k_b = qkv_b_r[:, 1, :].reshape(hidden)
        v_b = qkv_b_r[:, 2, :].reshape(hidden)
        assign([block_key, "attn", "q_proj", "bias"], q_b)
        assign([block_key, "attn", "k_proj", "bias"], k_b)
        assign([block_key, "attn", "v_proj", "bias"], v_b)

        o_w = state_dict[prefix + "attention.dense.weight"].cpu().float().numpy().T
        assign([block_key, "attn", "o_proj", "kernel"], o_w)
        if state_dict[prefix + "attention.dense.bias"] is not None:
            assign(
                [block_key, "attn", "o_proj", "bias"],
                state_dict[prefix + "attention.dense.bias"].cpu().float().numpy(),
            )

        # MLP
        logger.debug(f"  loading block {i} MLP weights...")
        h4_w = state_dict[prefix + "mlp.dense_h_to_4h.weight"].cpu().float().numpy().T
        h4_b = state_dict[prefix + "mlp.dense_h_to_4h.bias"].cpu().float().numpy()
        back_w = state_dict[prefix + "mlp.dense_4h_to_h.weight"].cpu().float().numpy().T
        back_b = state_dict[prefix + "mlp.dense_4h_to_h.bias"].cpu().float().numpy()
        assign([block_key, "mlp", "dense_h_to_4h", "kernel"], h4_w)
        assign([block_key, "mlp", "dense_h_to_4h", "bias"], h4_b)
        assign([block_key, "mlp", "dense_4h_to_h", "kernel"], back_w)
        assign([block_key, "mlp", "dense_4h_to_h", "bias"], back_b)

    # Final norm
    logger.debug("  loading final norm weights...")
    assign(
        ["ln_f", "weight"],
        state_dict["gpt_neox.final_layer_norm.weight"].cpu().float().numpy(),
    )
    if "gpt_neox.final_layer_norm.bias" in state_dict:
        assign(
            ["ln_f", "bias"],
            state_dict["gpt_neox.final_layer_norm.bias"].cpu().float().numpy(),
        )

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

    logger.debug("exporting embedding weights...")
    logger.debug("  loading embedding weights...")
    embed = torch.tensor(grab(["wte"]), dtype=torch.float32)
    head = torch.tensor(grab(["lm_head"]), dtype=torch.float32)
    sd["gpt_neox.embed_in.weight"] = embed
    sd["embed_out.weight"] = head

    hidden = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    head_dim = hidden // num_heads

    for i in range(n_layers):
        logger.debug(f"exporting block {i} weights...")
        prefix = f"gpt_neox.layers.{i}."
        block_key = f"blocks_{i}"

        logger.debug(f"  loading block {i} norm weights...")
        sd[prefix + "input_layernorm.weight"] = torch.tensor(
            grab([block_key, "ln_1", "weight"]), dtype=torch.float32
        )
        if "bias" in grab([block_key, "ln_1"]):
            sd[prefix + "input_layernorm.bias"] = torch.tensor(
                grab([block_key, "ln_1", "bias"]), dtype=torch.float32
            )
        logger.debug(f"  loading block {i} post-attention norm weights...")
        sd[prefix + "post_attention_layernorm.weight"] = torch.tensor(
            grab([block_key, "ln_2", "weight"]), dtype=torch.float32
        )
        if "bias" in grab([block_key, "ln_2"]):
            sd[prefix + "post_attention_layernorm.bias"] = torch.tensor(
                grab([block_key, "ln_2", "bias"]), dtype=torch.float32
            )

        # Re-interleave QKV weights per head for HF format
        logger.debug(f"  loading block {i} attention weights...")
        q_w = grab([block_key, "attn", "q_proj", "kernel"])  # (in, hidden)
        k_w = grab([block_key, "attn", "k_proj", "kernel"])
        v_w = grab([block_key, "attn", "v_proj", "kernel"])
        # Reshape to (in, num_heads, head_dim), stack as (in, num_heads, 3, head_dim)
        q_r = q_w.reshape(hidden, num_heads, head_dim)
        k_r = k_w.reshape(hidden, num_heads, head_dim)
        v_r = v_w.reshape(hidden, num_heads, head_dim)
        qkv = jnp.stack([q_r, k_r, v_r], axis=2).reshape(hidden, 3 * hidden)
        sd[prefix + "attention.query_key_value.weight"] = torch.tensor(
            qkv.T, dtype=torch.float32
        )

        q_b = grab([block_key, "attn", "q_proj", "bias"])
        k_b = grab([block_key, "attn", "k_proj", "bias"])
        v_b = grab([block_key, "attn", "v_proj", "bias"])
        q_b_r = q_b.reshape(num_heads, head_dim)
        k_b_r = k_b.reshape(num_heads, head_dim)
        v_b_r = v_b.reshape(num_heads, head_dim)
        qkv_b = jnp.stack([q_b_r, k_b_r, v_b_r], axis=1).reshape(3 * hidden)
        sd[prefix + "attention.query_key_value.bias"] = torch.tensor(
            qkv_b, dtype=torch.float32
        )

        o_w = grab([block_key, "attn", "o_proj", "kernel"])
        sd[prefix + "attention.dense.weight"] = torch.tensor(o_w.T, dtype=torch.float32)
        if "bias" in grab([block_key, "attn", "o_proj"]):
            sd[prefix + "attention.dense.bias"] = torch.tensor(
                grab([block_key, "attn", "o_proj", "bias"]), dtype=torch.float32
            )

        logger.debug(f"  loading block {i} MLP weights...")
        h4_w = grab([block_key, "mlp", "dense_h_to_4h", "kernel"])
        h4_b = grab([block_key, "mlp", "dense_h_to_4h", "bias"])
        back_w = grab([block_key, "mlp", "dense_4h_to_h", "kernel"])
        back_b = grab([block_key, "mlp", "dense_4h_to_h", "bias"])
        sd[prefix + "mlp.dense_h_to_4h.weight"] = torch.tensor(
            h4_w.T, dtype=torch.float32
        )
        sd[prefix + "mlp.dense_h_to_4h.bias"] = torch.tensor(h4_b, dtype=torch.float32)
        sd[prefix + "mlp.dense_4h_to_h.weight"] = torch.tensor(
            back_w.T, dtype=torch.float32
        )
        sd[prefix + "mlp.dense_4h_to_h.bias"] = torch.tensor(
            back_b, dtype=torch.float32
        )

    logger.debug("  loading final norm weights...")
    sd["gpt_neox.final_layer_norm.weight"] = torch.tensor(
        grab(["ln_f", "weight"]), dtype=torch.float32
    )
    if "bias" in grab(["ln_f"]):
        sd["gpt_neox.final_layer_norm.bias"] = torch.tensor(
            grab(["ln_f", "bias"]), dtype=torch.float32
        )
    logger.debug("export complete.")
    return sd


__all__ = ["GPTNeoX", "_from_hf_state_dict", "_to_hf_state_dict"]
