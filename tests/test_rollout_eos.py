import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from theseus.base import Axis
from theseus.inference.base import InferenceJob


@flax.struct.dataclass
class DummyState:
    params: jax.Array


class EOSRegressionInference(InferenceJob[object, object]):
    MODEL = object

    @staticmethod
    def forward(
        state,
        params,
        batch,
        key=None,
        deterministic=False,
        mutable=None,
        extra_variables=None,
        cache_max_len=None,
    ):
        del state, params, key, deterministic, extra_variables, cache_max_len
        x_batch, _, _ = batch
        vocab_size = 16

        if x_batch.shape[1] > 1:
            next_token = jnp.full((x_batch.shape[0],), 9, dtype=jnp.int32)
        else:
            last_token = x_batch[:, 0]
            next_token = jnp.where(
                last_token == 9,
                7,
                jnp.where(last_token == 7, 8, 6),
            ).astype(jnp.int32)

        logits = jax.nn.one_hot(
            next_token, vocab_size, dtype=jnp.float32
        )[:, None, :] * 100.0
        logits = jnp.broadcast_to(
            logits, (x_batch.shape[0], x_batch.shape[1], vocab_size)
        )

        if mutable is not None:
            return (logits, None, {}), {}
        return logits, None, {}


def _build_inference() -> EOSRegressionInference:
    inference = object.__new__(EOSRegressionInference)
    inference._trainer = None
    inference.replicas = 1
    inference.local_replicas = 1
    inference.per_device_batch_size = 1
    inference.block_size = 8
    inference.mesh = Mesh(
        np.array(jax.devices()).reshape((1, 1)), (Axis.BATCH, Axis.SHARD)
    )
    scalar_sharding = NamedSharding(inference.mesh, P())
    inference.state = DummyState(
        params=jax.device_put(jnp.array(0, dtype=jnp.int32), scalar_sharding)
    )
    inference.key = jax.random.PRNGKey(0)
    return inference


def test_autoregress_does_not_force_eos_after_first_eos():
    inference = _build_inference()
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    input_mask = jnp.array([[True, True, True]], dtype=jnp.bool_)

    result = inference._autoregress(
        inference.state,
        jax.random.PRNGKey(0),
        input_ids,
        input_mask,
        num_tokens=6,
        temperature=0.0,
        top_p=1.0,
        eos_token=9,
    )

    assert result.tolist()[0] == [1, 2, 3, 9, 7, 8]
