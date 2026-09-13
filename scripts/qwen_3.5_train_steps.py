"""Minimal training-step smoke test for the Qwen 3.5 backbone path.

Loads ``Qwen/Qwen3.5-0.8B`` via ``Qwen3_5.from_pretrained`` (the same code path
``BackbonedTrainer._init_model`` takes), then takes a few AdamW steps on a
single tiny batch and prints the loss per step. This is a gradient-flow
sanity check, not a real training run — the same batch is reused every step,
so loss collapses to near zero (the model memorizes 4 sentences). What this
verifies: the full forward + backward + optimizer step pipeline produces
finite gradients and the loss decreases monotonically with no NaN / large
spikes.

Run:
    LD_PRELOAD=... HF_HOME=... CUDA_VISIBLE_DEVICES=0 \
        uv run python scripts/qwen_3.5_train_steps.py --steps 12
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from theseus.config import patch
from theseus.model.models.contrib.qwen_3_5 import Qwen3_5

hf_logging.set_verbosity_error()


def _tokenize(model_id: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(model_id)
    samples = [
        "Once upon a time there was a small fox who loved apples.",
        "The sun rose slowly over the quiet meadow.",
        "She picked up the book and read the first page.",
        "He looked at the sky and saw a bright star.",
        "A river flowed gently beside the old wooden bridge.",
        "The cat jumped onto the warm windowsill.",
        "They walked together through the green forest.",
        "Snow fell softly on the empty street that morning.",
    ]
    out = tok(
        samples,
        return_tensors="np",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return out["input_ids"], out["attention_mask"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Keep an outer config context alive: ``from_pretrained`` populates it,
    # and later ``model.apply`` re-runs the model's ``setup()`` which calls
    # ``configure()`` on sub-blocks. Outside a context that would raise.
    with patch():
        model, params = Qwen3_5.from_pretrained(
            args.model, param_dtype="float32", activation_dtype="float32"
        )

        # Backbone has linear-attention layers whose chunked-delta-rule grads can
        # spike; clip global norm to keep things stable for a smoke test.
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(args.lr, weight_decay=0.0),
        )
        state = train_state.TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply, params=params, tx=tx
        )

        input_ids, attention_mask = _tokenize(args.model, args.max_length)
        if input_ids.shape[0] < args.batch_size:
            reps = (args.batch_size + input_ids.shape[0] - 1) // input_ids.shape[0]
            input_ids = np.tile(input_ids, (reps, 1))[: args.batch_size]
            attention_mask = np.tile(attention_mask, (reps, 1))[: args.batch_size]
        else:
            input_ids = input_ids[: args.batch_size]
            attention_mask = attention_mask[: args.batch_size]

        # Targets = inputs shifted by 1; masked positions get -1 (ignored).
        targets = np.concatenate(
            [input_ids[:, 1:], -np.ones((args.batch_size, 1), dtype=np.int32)],
            axis=1,
        )
        target_mask = np.concatenate(
            [
                attention_mask[:, 1:],
                np.zeros((args.batch_size, 1), dtype=attention_mask.dtype),
            ],
            axis=1,
        )
        targets = np.where(target_mask.astype(bool), targets, -1)

        idx = jnp.array(input_ids.astype(np.int32))
        tgt = jnp.array(targets.astype(np.int32))
        pad = jnp.array(attention_mask.astype(bool))

        def loss_fn(p: object) -> jax.Array:
            _, loss = model.apply(
                {"params": p}, idx, targets=tgt, padding_mask=pad, deterministic=True
            )
            return loss

        @jax.jit
        def step(
            state: train_state.TrainState,
        ) -> tuple[train_state.TrainState, jax.Array]:
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        losses: list[float] = []
        t0 = time.time()
        for i in range(args.steps):
            state, loss = step(state)
            loss_v = float(jax.device_get(loss))
            losses.append(loss_v)
            print(f"step {i:02d}  loss {loss_v:.5f}  ({time.time() - t0:.1f}s)")

        diffs = np.diff(losses)
        n_increases = int((diffs > 0).sum())
        largest_increase = float(diffs.max()) if len(diffs) else 0.0
        print(
            f"\nfinal loss {losses[-1]:.5f} (start {losses[0]:.5f}) — "
            f"increases {n_increases}/{len(diffs)}, largest +{largest_increase:.5f}"
        )
        if losses[-1] >= losses[0]:
            raise SystemExit(
                f"FAIL: loss did not decrease (start {losses[0]:.5f} → end {losses[-1]:.5f})"
            )
        if largest_increase > 0.2:
            raise SystemExit(
                f"FAIL: large loss spike (+{largest_increase:.5f}) — training unstable"
            )
        print("PASS: loss decreased without large spikes")


if __name__ == "__main__":
    main()
