"""Output projections use compute precision; cross-entropy uses FP32."""

import jax
import jax.numpy as jnp
import pytest

from theseus.config import build, configuration, configure
from theseus.model.models.contrib.gpt_neox import GPTNeoX
from theseus.model.models.contrib.llama import Llama
from theseus.model.models.contrib.marin import Marin
from theseus.model.models.contrib.qwen import Qwen
from theseus.model.models.contrib.qwen_3_5 import Qwen3_5


@pytest.mark.parametrize("model_type", [GPTNeoX, Llama, Marin, Qwen, Qwen3_5])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_unembedding_and_loss_precision(model_type, dtype):
    config = build(*model_type.gather())
    config.architecture.n_layers = 0
    config.architecture.n_embd = 8
    config.architecture.vocab_size = 16
    config.architecture.rms_norm_eps = 1e-6
    config.architecture.dtype.param = "float32"
    config.architecture.dtype.activation = dtype
    x = jnp.ones((1, 2, 8), dtype=jnp.float32)
    targets = jnp.zeros((1, 2), dtype=jnp.int32)
    with configuration(config):
        model = configure(model_type)
        variables = model.init(jax.random.key(0), x, method=model.unembed)

        def loss(params):
            logits = model.apply({"params": params}, x, method=model.unembed)
            assert logits.dtype == jnp.dtype(dtype)
            return model.apply({"params": params}, logits, targets, method=model.loss)

        params = jax.tree.map(lambda p: p.astype(dtype), variables["params"])
        value, grads = jax.value_and_grad(loss)(params)
        assert value.dtype == jnp.float32
        assert jnp.isfinite(value)
        assert all(p.dtype == jnp.dtype(dtype) for p in jax.tree.leaves(grads))
        dots = [
            eqn for eqn in jax.make_jaxpr(loss)(params).jaxpr.eqns
            if eqn.primitive.name == "dot_general"
        ]
        assert len(dots) == 1
        assert all(v.aval.dtype == jnp.dtype(dtype) for v in dots[0].invars)
