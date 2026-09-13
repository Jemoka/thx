"""Smoke test: init from HF checkpoint via the configure/patch refactor.

Verifies that from_pretrained loads weights and produces a valid forward pass
for each supported architecture.

Usage:
    uv run python scripts/smoke_test_from_pretrained.py
    uv run python scripts/smoke_test_from_pretrained.py --models pythia tinyllama qwen
"""

import argparse
import sys

import jax.numpy as jnp
from transformers.utils import logging as hf_logging

from theseus.config import patch

hf_logging.set_verbosity_error()

MODELS = {
    "pythia": "EleutherAI/pythia-70m-deduped",  # small, fast
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "qwen": "Qwen/Qwen2.5-0.5B",
}


def test_pythia(model_id: str) -> None:
    from theseus.model.models.contrib.gpt_neox import GPTNeoX

    with patch():
        model, params = GPTNeoX.from_pretrained(model_id)
        dummy = jnp.array([[1, 2, 3, 4, 5]])
        logits, _ = model.apply({"params": params}, dummy, deterministic=True)

    assert logits.shape == (1, 5, model.vocab_size), f"unexpected shape {logits.shape}"
    print(
        f"  pythia  ok — {model.n_layers}L n_embd={model.n_embd} logits={logits.shape}"
    )


def test_tinyllama(model_id: str) -> None:
    from theseus.model.models.contrib.llama import Llama

    with patch():
        model, params = Llama.from_pretrained(model_id)
        dummy = jnp.array([[1, 2, 3, 4, 5]])
        logits, _ = model.apply({"params": params}, dummy, deterministic=True)

    assert logits.shape == (1, 5, model.vocab_size), f"unexpected shape {logits.shape}"
    print(
        f"  llama   ok — {model.n_layers}L n_embd={model.n_embd} logits={logits.shape}"
    )


def test_qwen(model_id: str) -> None:
    from theseus.model.models.contrib.qwen import Qwen

    with patch():
        model, params = Qwen.from_pretrained(model_id)
        dummy = jnp.array([[1, 2, 3, 4, 5]])
        logits, _ = model.apply({"params": params}, dummy, deterministic=True)

    assert logits.shape == (1, 5, model.vocab_size), f"unexpected shape {logits.shape}"
    print(
        f"  qwen    ok — {model.n_layers}L n_embd={model.n_embd} logits={logits.shape}"
    )


RUNNERS = {
    "pythia": test_pythia,
    "tinyllama": test_tinyllama,
    "qwen": test_qwen,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="which models to test (default: all)",
    )
    for key in MODELS:
        parser.add_argument(f"--{key}", dest=f"model_{key}", default=MODELS[key])
    args = parser.parse_args()

    overrides = {k: getattr(args, f"model_{k}") for k in MODELS}

    failed = []
    for name in args.models:
        model_id = overrides[name]
        print(f"[{name}] {model_id}")
        try:
            RUNNERS[name](model_id)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nall ok")


if __name__ == "__main__":
    main()
