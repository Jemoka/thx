import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from theseus.base import Axis
from theseus.data.datasets import ChatTurn
from theseus.data.tokenizer import HuggingFaceTokenizer
from theseus.inference.base import InferenceJob


@flax.struct.dataclass
class DummyState:
    params: jax.Array


class DummyInference:
    pass


class FakeHFBackend:
    eos_token_id = 0

    def apply_chat_template(
        self,
        turns,
        tokenize,
        add_generation_prompt,
        return_tensors=None,
    ):
        del return_tensors
        user_msg = turns[-1]["content"]
        if tokenize:
            if user_msg == "What is the capital of France?":
                return [11, 12]
            if user_msg == "Fuck you!":
                return [21, 22]
            raise AssertionError(f"unexpected user message: {user_msg}")
        suffix = "|assistant" if add_generation_prompt else ""
        return f"rendered::{user_msg}{suffix}"

    def __call__(self, texts, add_special_tokens=False):
        del add_special_tokens
        # If rollout ever falls back to render-then-batch-tokenize for chat
        # inputs, this intentionally corrupts the second prompt.
        ids = []
        for text in texts:
            if "Fuck you!" in text:
                ids.append([])
            else:
                ids.append([31, 32])
        return {"input_ids": ids}

    def decode(
        self,
        tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(tok) for tok in tokens)

    def batch_decode(
        self,
        tokens_batch,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return [self.decode(tokens) for tokens in tokens_batch]


def _build_inference() -> DummyInference:
    inference = DummyInference()
    inference.replicas = 1
    inference.local_replicas = 1
    inference.per_device_batch_size = 2
    inference.block_size = 8
    inference.mesh = Mesh(np.array(jax.devices()).reshape((1,)), (Axis.BATCH,))
    scalar_sharding = NamedSharding(inference.mesh, P())
    inference.state = DummyState(
        params=jax.device_put(jnp.array(0, dtype=jnp.int32), scalar_sharding)
    )
    inference.state_sharding = DummyState(params=scalar_sharding)
    inference.key = jax.random.PRNGKey(0)
    inference._rollout_chunk_jit = None
    inference._rollout_chunk_jit_key = None
    inference.pad = staticmethod(InferenceJob.pad)
    inference._get_rollout_chunk_jit = InferenceJob._get_rollout_chunk_jit.__get__(
        inference, DummyInference
    )
    inference.rollout = InferenceJob.rollout.__get__(inference, DummyInference)

    def _autoregress(
        state,
        key,
        input,
        input_mask,
        num_tokens,
        temperature,
        top_p,
        eos_token=None,
    ):
        del state, key, input_mask, temperature, top_p, eos_token
        extra = num_tokens - input.shape[-1]
        if extra <= 0:
            return input[:, :num_tokens]
        generated = jnp.full((input.shape[0], extra), 777, dtype=jnp.int32)
        return jnp.concatenate([input, generated], axis=-1)

    inference._autoregress = _autoregress
    return inference


def test_rollout_chat_inputs_use_direct_hf_chat_tokenization():
    inference = _build_inference()
    tokenizer = HuggingFaceTokenizer(FakeHFBackend())
    convos = [
        [
            ChatTurn("system", "You are a helpful assistant."),
            ChatTurn("user", "What is the capital of France?"),
        ],
        [
            ChatTurn("system", "You are a helpful assistant."),
            ChatTurn("user", "Fuck you!"),
        ],
    ]

    out = inference.rollout(
        convos,
        encoding=tokenizer,
        max_new_tokens=2,
        max_prompt_length=4,
        return_type="indices",
    )

    assert out == [[11, 12, 777, 777], [21, 22, 777, 777]]
