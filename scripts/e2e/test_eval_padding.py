#!/usr/bin/env python3
"""Exercise evaluation padding when dataset size is smaller than the batch unit."""

from __future__ import annotations


import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from theseus.base import Axis
from theseus.evaluation import (
    EncodingEvaluation,
    PerplexityComparisonEvaluation,
    PerplexityEvaluation,
    RolloutEvaluation,
)


class ToyTokenizer:
    pad_token = 0

    def encode(self, text: str) -> list[int]:
        return [ord(ch) + 1 for ch in text]

    def encode_batch(
        self, text_list: list[str], allowed_special: str | None = None
    ) -> list[list[int]]:
        del allowed_special
        return [self.encode(text) for text in text_list]

    def decode_batch(self, seqs: list[list[int]]) -> list[str]:
        decoded: list[str] = []
        for seq in seqs:
            decoded.append("".join(chr(tok - 1) for tok in seq if tok > 0))
        return decoded


@flax.struct.dataclass
class DummyState:
    params: jax.Array


class DummyInference:
    def __init__(
        self, batch_unit: int = 8, block_size: int = 8, rollout_token: str = "!"
    ):
        if jax.process_count() != 1:
            raise RuntimeError("This script expects a single-process JAX runtime.")

        self.replicas = 1
        self.local_replicas = 1
        self.per_device_batch_size = batch_unit
        self.block_size = block_size
        self.mesh = Mesh(np.array(jax.devices()).reshape((1,)), (Axis.BATCH,))
        scalar_sharding = NamedSharding(self.mesh, P())
        self.state = DummyState(
            params=jax.device_put(jnp.array(0, dtype=jnp.int32), scalar_sharding)
        )
        self.state_sharding = DummyState(params=scalar_sharding)
        self.key = jax.random.PRNGKey(0)
        self.vocab_size = 256
        self.rollout_token_id = ord(rollout_token) + 1

    @staticmethod
    def pad(seqs: list[list[int]], pad_token: int = 0) -> tuple[jax.Array, jax.Array]:
        max_len = max(len(seq) for seq in seqs)
        padded = [([pad_token] * (max_len - len(seq))) + seq for seq in seqs]
        masks = [([False] * (max_len - len(seq))) + ([True] * len(seq)) for seq in seqs]
        return jnp.array(padded, dtype=jnp.int32), jnp.array(masks, dtype=jnp.bool_)

    def _autoregress(
        self,
        state: DummyState,
        key: jax.Array,
        input: jax.Array,
        input_mask: jax.Array,
        num_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jax.Array:
        del state, key, input_mask, temperature, top_p
        extra = num_tokens - input.shape[-1]
        if extra <= 0:
            return input[:, :num_tokens]
        generated = jnp.full(
            (input.shape[0], extra), self.rollout_token_id, dtype=jnp.int32
        )
        return jnp.concatenate([input, generated], axis=-1)

    def forward(
        self,
        state: DummyState,
        params: jax.Array,
        batch: tuple[jax.Array, None, jax.Array],
        key: jax.Array | None,
        deterministic: bool,
    ) -> tuple[jax.Array, None, None]:
        del state, params, key, deterministic
        x_batch, _, mask_batch = batch
        next_tokens = jnp.roll(x_batch, -1, axis=-1)
        next_tokens = next_tokens.at[:, -1].set(0)
        next_tokens = jnp.where(mask_batch, next_tokens, 0)
        logits = jax.nn.one_hot(next_tokens, self.vocab_size, dtype=jnp.float32) * 20.0
        return logits, None, None


class ToyRolloutEval(RolloutEvaluation):
    def __init__(self) -> None:
        self.items = [("aa", "!"), ("bb", "!"), ("cc", "!")]

    @property
    def name(self) -> str:
        return "toy_rollout"

    def __len__(self) -> int:
        return len(self.items)

    def clean(self, y_hat: str) -> str:
        return y_hat[-1:]

    def get(self, indx: int) -> tuple[str, str]:
        return self.items[indx]

    def max_new_tokens(self, inference: DummyInference) -> int:
        del inference
        return 1

    def check(self, y: str, y_hat: str) -> bool:
        return y == y_hat


class ToyEncodingEval(EncodingEvaluation):
    def __init__(self) -> None:
        self.items = ["abc", "def", "ghi"]

    @property
    def name(self) -> str:
        return "toy_encoding"

    def __len__(self) -> int:
        return len(self.items)

    def clean(self, y_hat: str) -> str:
        return y_hat

    def get(self, indx: int) -> str:
        return self.items[indx]

    def check(self, x: str, y_hat: str) -> bool:
        return y_hat == x[1:]


class ToyPerplexityEval(PerplexityEvaluation):
    def __init__(self) -> None:
        self.items = ["abc", "def", "ghi"]

    @property
    def name(self) -> str:
        return "toy_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]


class ToyComparisonEval(PerplexityComparisonEvaluation):
    def __init__(self) -> None:
        self.items = [
            ("a", ["bc", "zz"], 0),
            ("d", ["ef", "yy"], 0),
            ("g", ["hi", "xx"], 0),
        ]

    @property
    def name(self) -> str:
        return "toy_compare"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> tuple[str, list[str], int]:
        return self.items[indx]


def main() -> None:
    tokenizer = ToyTokenizer()
    inference = DummyInference(batch_unit=8, block_size=8)

    rollout_score = ToyRolloutEval()(inference, tokenizer)
    encoding_score = ToyEncodingEval()(inference, tokenizer)
    ppl_score = ToyPerplexityEval()(inference, tokenizer)
    comparison_score = ToyComparisonEval()(inference, tokenizer)

    print(f"batch_unit={inference.replicas * inference.per_device_batch_size}")
    print("dataset_size=3, flattened_comparison_size=6")
    print(f"rollout_score={rollout_score:.6f}")
    print(f"encoding_score={encoding_score:.6f}")
    print(f"ppl_score={ppl_score:.6f}")
    print(f"comparison_score={comparison_score:.6f}")

    assert rollout_score == 1.0
    assert encoding_score == 1.0
    assert 0.99 < ppl_score <= 1.0
    assert comparison_score == 1.0

    print("evaluation padding test passed")


if __name__ == "__main__":
    main()
