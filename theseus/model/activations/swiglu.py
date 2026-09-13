import jax
import jax.numpy as jnp


def swiglu(gate: jnp.ndarray, up: jnp.ndarray) -> jnp.ndarray:
    """SwiGLU activation: silu(gate) * up."""
    return jnp.multiply(jax.nn.silu(gate), up)
