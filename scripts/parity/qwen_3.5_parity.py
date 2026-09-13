"""HF<->JAX Qwen 3.5 (text-only) parity check.

Usage: uv run python scripts/qwen_3.5_parity.py --model Qwen/Qwen3.5-0.8B --prompt "Hello world"

Note: the dense Qwen 3.5 checkpoint carries vision (`visual.*`) keys. The
state-dict loader must tolerate / skip them so init isn't broken when we only
care about the text path.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.utils import logging as hf_logging

from theseus.config import patch, configure
from theseus.model.models.contrib.qwen_3_5 import (
    Qwen3_5,
    _from_hf_state_dict,
    _to_hf_state_dict,
)

hf_logging.set_verbosity_error()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

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
        outputs_hf = hf(**inputs)
        logits_hf = outputs_hf.logits.detach().cpu().numpy()

    # Qwen3.5ForCausalLM wraps a text config; HF nests it under .config.text_config
    # for the full multimodal config, but the dense LM-only weights are loadable
    # via either path. Prefer .text_config when present.
    cfg = getattr(hf.config, "text_config", hf.config)
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", 10000.0)
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)

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
                "dtype": {"param": "float32", "activation": "float32"},
            }
        )

        model = configure(Qwen3_5)
        dummy = jnp.zeros((1, 1), dtype=jnp.int32)
        abstract = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
        params = jax.tree_util.tree_map(
            lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
        )
        layer_types = list(cfg.layer_types)
        params = _from_hf_state_dict(params, hf.state_dict(), layer_types)

        idx = jnp.array(inputs["input_ids"].numpy())
        attn_bool = jnp.array(inputs["attention_mask"].numpy(), dtype=bool)
        logits_jax, _ = model.apply(
            {"params": params}, idx, padding_mask=attn_bool, deterministic=True
        )

        attn = inputs["attention_mask"][0].numpy()
        last_tok_idx = int(attn.sum() - 1)
        logits_hf_last = logits_hf[0, last_tok_idx]
        logits_jax_last = np.array(logits_jax[0, last_tok_idx])
        max_diff = np.max(np.abs(logits_hf_last - logits_jax_last))
        mean_diff = np.mean(np.abs(logits_hf_last - logits_jax_last))
        overlap = len(
            set(logits_hf_last.argsort()[-5:]) & set(logits_jax_last.argsort()[-5:])
        )

        print(f"max diff: {max_diff}")
        print(f"mean diff: {mean_diff}")
        print(f"top5 overlap: {overlap}")

        sd = _to_hf_state_dict(params, layer_types)
        assert sd["model.embed_tokens.weight"].shape[0] == cfg.vocab_size
        print("export state_dict keys:", len(sd))

        hf_rt = Qwen3_5ForCausalLM(hf.config)
        hf_rt.load_state_dict(sd, strict=False)
        hf_rt.eval()
        with torch.no_grad():
            logits_hf_rt = hf_rt(**inputs).logits.detach().cpu().numpy()
        rt_max = np.max(np.abs(logits_hf - logits_hf_rt))
        rt_mean = np.mean(np.abs(logits_hf - logits_hf_rt))
        print(f"roundtrip hf->jax->hf max diff: {rt_max}")
        print(f"roundtrip hf->jax->hf mean diff: {rt_mean}")

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        with torch.no_grad():
            loss_hf = hf(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            ).loss.item()

        idx_np = inputs["input_ids"].numpy()
        attn_np = inputs["attention_mask"].numpy().astype(bool)
        logits_jax_full, _ = model.apply(
            {"params": params},
            jnp.array(idx_np),
            padding_mask=jnp.array(attn_np),
            deterministic=True,
        )
        logits_jax_np = np.array(logits_jax_full)
        logits_shift = logits_jax_np[:, :-1, :]
        targets = idx_np[:, 1:]
        mask = attn_np[:, 1:]
        log_probs = logits_shift - np.log(np.exp(logits_shift).sum(-1, keepdims=True))
        nll = -np.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        nll = nll * mask
        loss_jax = nll.sum() / mask.sum()
        print(f"hf loss: {loss_hf}")
        print(f"jax loss: {loss_jax}")


if __name__ == "__main__":
    main()
