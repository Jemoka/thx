"""
A theseus nn module, which is nn module, execpt it asks you to
dump all the types of constitutent parts so we can configure it.
"""

from typing import Type, List, Any
from flax.linen import Module as nn
import jax.numpy as jnp

from theseus.base.axis import ShardingPlan
from theseus.config import field

DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


class Module(nn):
    param_dtype: str = field("architecture/dtype/param", default="float32")
    activation_dtype: str = field("architecture/dtype/activation", default="bfloat16")

    def flops(self, seq: int) -> float:
        """Estimated forward + backward FLOPs for one sequence of length ``seq``.

        Composite modules require binding to existing variables so their actual
        setup children are available. Counts use dense matmul arithmetic (one
        multiply-add is two FLOPs, backward is twice forward), excluding
        optimizer updates, rematerialization and minor elementwise operations.
        Unsupported operations contribute zero; this is not an exact counter.
        """
        return 0.0

    @staticmethod
    def _parse_dtype(s: str) -> Any:
        if s not in DTYPE_MAP:
            raise ValueError(f"Unknown dtype {s!r}, expected one of {list(DTYPE_MAP)}")
        return DTYPE_MAP[s]

    @property
    def _param_dtype(self) -> Any:
        return self._parse_dtype(self.param_dtype)

    @property
    def _activation_dtype(self) -> Any:
        return self._parse_dtype(self.activation_dtype)

    @property
    def sharding(self) -> ShardingPlan:
        """Logical mappings; models without a plan use replicated parameters."""
        return ShardingPlan()

    @classmethod
    def components(cls) -> List[Type[Any]]:
        """Return the types of constituent parts of this module.

        Returns:
            Type: A type or tuple of types representing the constituent parts.
        """

        raise NotImplementedError(
            "To configure a module, you must implement components()"
        )

    @classmethod
    def gather(cls) -> List[Type[Any]]:
        """Depth-first search of all constituent parts of this module.

        Returns:
            List[Type]: A list of all constituent part types.
        """

        parts: List[Type[Any]] = [cls]
        for component in cls.components():
            if issubclass(component, Module):
                parts.extend(component.gather())
            else:
                parts.append(component)
        return list(set(parts))
