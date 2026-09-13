"""Evaluation padding tests — verifies evaluations work when dataset
size is not a multiple of the batch unit.

Migrated from scripts/test_eval_padding.py.
"""

import pytest
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from types import SimpleNamespace
from theseus.base import Axis, ShardingPlan
from theseus.evaluation import (
    BitsPerByteEvaluation,
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
        return ["".join(chr(tok - 1) for tok in seq if tok > 0) for seq in seqs]


@flax.struct.dataclass
class DummyState:
    params: jax.Array


class DummyInference:
    def __init__(
        self, batch_unit: int = 8, block_size: int = 8, rollout_token: str = "!"
    ):
        self.model = SimpleNamespace(sharding=ShardingPlan())
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

    def rollout(self, inputs, **kwargs):
        return [
            [0] * (kwargs["max_prompt_length"] - len(tokens))
            + list(tokens)
            + [self.rollout_token_id] * kwargs["max_new_tokens"]
            for tokens in inputs
        ]

    @staticmethod
    def pad(seqs: list[list[int]], pad_token: int = 0) -> tuple[jax.Array, jax.Array]:
        max_len = max(len(seq) for seq in seqs)
        padded = [([pad_token] * (max_len - len(seq))) + seq for seq in seqs]
        masks = [([False] * (max_len - len(seq))) + ([True] * len(seq)) for seq in seqs]
        return jnp.array(padded, dtype=jnp.int32), jnp.array(masks, dtype=jnp.bool_)

    def _autoregress(
        self, state, key, input, input_mask, num_tokens, temperature, top_p
    ):
        del state, key, input_mask, temperature, top_p
        extra = num_tokens - input.shape[-1]
        if extra <= 0:
            return input[:, :num_tokens]
        generated = jnp.full(
            (input.shape[0], extra), self.rollout_token_id, dtype=jnp.int32
        )
        return jnp.concatenate([input, generated], axis=-1)

    def forward(self, state, params, batch, key, deterministic):
        del state, params, key, deterministic
        x_batch, _, mask_batch = batch
        next_tokens = jnp.roll(x_batch, -1, axis=-1)
        next_tokens = next_tokens.at[:, -1].set(0)
        next_tokens = jnp.where(mask_batch, next_tokens, 0)
        logits = jax.nn.one_hot(next_tokens, self.vocab_size, dtype=jnp.float32) * 20.0
        return logits, None, None


class ToyRolloutEval(RolloutEvaluation):
    def __init__(self):
        self.items = [("aa", "!"), ("bb", "!"), ("cc", "!")]

    @property
    def name(self):
        return "toy_rollout"

    def __len__(self):
        return len(self.items)

    def clean(self, y_hat):
        return y_hat[-1:]

    def get(self, indx):
        return self.items[indx]

    def max_new_tokens(self, inference):
        return 1

    def check(self, y, y_hat):
        return y == y_hat


class ToyEncodingEval(EncodingEvaluation):
    def __init__(self):
        self.items = ["abc", "def", "ghi"]

    @property
    def name(self):
        return "toy_encoding"

    def __len__(self):
        return len(self.items)

    def clean(self, y_hat):
        return y_hat

    def get(self, indx):
        return self.items[indx]

    def check(self, x, y_hat):
        return y_hat == x[1:]


class ToyPerplexityEval(PerplexityEvaluation):
    def __init__(self):
        self.items = ["abc", "def", "ghi"]

    @property
    def name(self):
        return "toy_ppl"

    def __len__(self):
        return len(self.items)

    def get(self, indx):
        return self.items[indx]


class ToyBitsPerByteEval(BitsPerByteEvaluation):
    def __init__(self):
        self.items = ["abc", "def", "ghi"]

    @property
    def name(self):
        return "toy_bpb"

    def __len__(self):
        return len(self.items)

    def get(self, indx):
        return self.items[indx]


class ToyComparisonEval(PerplexityComparisonEvaluation):
    def __init__(self):
        self.items = [
            ("a", ["bc", "zz"], 0),
            ("d", ["ef", "yy"], 0),
            ("g", ["hi", "xx"], 0),
        ]

    @property
    def name(self):
        return "toy_compare"

    def __len__(self):
        return len(self.items)

    def get(self, indx):
        return self.items[indx]


class TestEvalPadding:
    """Test that evaluations handle padding correctly when dataset size < batch unit."""

    def test_rollout_eval(self):
        tokenizer = ToyTokenizer()
        inference = DummyInference(batch_unit=8, block_size=8)
        score = ToyRolloutEval()(inference, tokenizer)
        assert score == 1.0

    def test_encoding_eval(self):
        tokenizer = ToyTokenizer()
        inference = DummyInference(batch_unit=8, block_size=8)
        score = ToyEncodingEval()(inference, tokenizer)
        assert score == 1.0

    def test_perplexity_eval(self):
        tokenizer = ToyTokenizer()
        inference = DummyInference(batch_unit=8, block_size=8)
        score = ToyPerplexityEval()(inference, tokenizer)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_bits_per_byte_eval(self):
        tokenizer = ToyTokenizer()
        inference = DummyInference(batch_unit=8, block_size=8)
        score = ToyBitsPerByteEval()(inference, tokenizer)
        assert 0.0 <= score < 1e-4

    def test_bits_per_byte_formula(self):
        nll = np.array([2.0 * np.log(2.0), 4.0 * np.log(2.0)])
        token_count = np.array([2.0, 4.0])

        eval_data = ToyBitsPerByteEval()

        per_sample = eval_data.score(nll, token_count, np.array([4.0, 2.0]))
        np.testing.assert_allclose(per_sample, np.array([0.5, 2.0]), rtol=1e-6)

        score = eval_data._score_from_stats(nll, token_count, ["abcd", "xy"])
        assert score == 1.0

    def test_comparison_eval(self):
        tokenizer = ToyTokenizer()
        inference = DummyInference(batch_unit=8, block_size=8)
        score = ToyComparisonEval()(inference, tokenizer)
        assert score == 1.0
