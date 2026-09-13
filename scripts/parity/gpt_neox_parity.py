"""HF<->JAX GPT-NeoX parity check.

Usage: uv run python scripts/gpt_neox_parity.py --model EleutherAI/pythia-70m-deduped --prompt "Hello"
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.utils import logging as hf_logging
from transformers.masking_utils import create_causal_mask

from theseus.model.models.contrib.gpt_neox import (
    GPTNeoX,
    _from_hf_state_dict,
    _to_hf_state_dict,
)

hf_logging.set_verbosity_error()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    hf = GPTNeoXForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=None
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    inputs = tok(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )

    with torch.no_grad():
        outputs_hf = hf(**inputs)
        logits_hf = outputs_hf.logits.detach().cpu().numpy()

    cfg = hf.config
    rope_theta = getattr(cfg, "rotary_emb_base", 10000.0)
    partial_rotary_factor = getattr(cfg, "rotary_pct", 1.0)
    if cfg.rope_parameters:
        rope_theta = cfg.rope_parameters.get("rope_theta", rope_theta)
        partial_rotary_factor = cfg.rope_parameters.get(
            "partial_rotary_factor", partial_rotary_factor
        )
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    model = GPTNeoX(
        n_layers=cfg.num_hidden_layers,
        n_embd=cfg.hidden_size,
        n_head=cfg.num_attention_heads,
        n_kv_head=n_kv,
        intermediate_size=cfg.intermediate_size,
        block_size=cfg.max_position_embeddings,
        vocab_size=cfg.vocab_size,
        dropout=cfg.hidden_dropout,
        attn_dropout=cfg.attention_dropout,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        layer_norm_eps=cfg.layer_norm_eps,
        bias=True,
        attention_bias=cfg.attention_bias if hasattr(cfg, "attention_bias") else True,
        use_parallel_residual=cfg.use_parallel_residual,
    )

    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 1), dtype=jnp.int32))[
        "params"
    ]
    params = _from_hf_state_dict(params, hf.state_dict(), cfg.num_hidden_layers, cfg)

    idx = jnp.array(inputs["input_ids"].numpy())
    # Use the exact HF causal mask (bool, True=keep) to avoid any semantic drift
    hf_causal = create_causal_mask(
        hf.config,
        hf.gpt_neox.embed_in(inputs["input_ids"]),
        inputs["attention_mask"],
        torch.arange(inputs["input_ids"].shape[1]),
        None,
        None,
    )
    logits_jax, _ = model.apply(
        {"params": params},
        idx,
        padding_mask=jnp.array(hf_causal.numpy(), dtype=bool),
        deterministic=True,
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

    sd = _to_hf_state_dict(params, cfg.num_hidden_layers, cfg)
    print("export state_dict keys:", len(sd))

    hf_rt = GPTNeoXForCausalLM(cfg)
    hf_rt.load_state_dict(sd, strict=True)
    hf_rt.eval()
    with torch.no_grad():
        logits_hf_rt = hf_rt(**inputs).logits.detach().cpu().numpy()
    rt_max = np.max(np.abs(logits_hf - logits_hf_rt))
    rt_mean = np.mean(np.abs(logits_hf - logits_hf_rt))
    print(f"roundtrip hf->jax->hf max diff: {rt_max}")
    print(f"roundtrip hf->jax->hf mean diff: {rt_mean}")

    # Cross-entropy loss comparison (causal shift, ignore padding)
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
        padding_mask=jnp.array(hf_causal.numpy(), dtype=bool),
        deterministic=True,
    )
    logits_jax_np = np.array(logits_jax_full)
    logits_shift = logits_jax_np[:, :-1, :]
    targets = idx_np[:, 1:]
    mask = attn_np[:, 1:]
    # stable log softmax
    logits_shift = logits_shift - logits_shift.max(axis=-1, keepdims=True)
    log_probs = logits_shift - np.log(np.exp(logits_shift).sum(-1, keepdims=True))
    nll = -np.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    nll = nll * mask
    loss_jax = nll.sum() / mask.sum()
    print(f"hf loss: {loss_hf}")
    print(f"jax loss: {loss_jax}")


if __name__ == "__main__":
    main()
