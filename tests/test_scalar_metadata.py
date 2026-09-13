from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from theseus.training.base import BaseTrainer
from theseus.training.ppo import PPOTrainer
from theseus.training.utils import scalar_metadata


class RecordingApply:
    def __init__(self) -> None:
        self.mutable_calls: list[Any] = []

    def __call__(
        self,
        variables: dict[str, Any],
        x: jnp.ndarray,
        targets: jnp.ndarray,
        *,
        mutable: Any = None,
        **kwargs: Any,
    ) -> Any:
        del variables, targets, kwargs
        self.mutable_calls.append(mutable)
        logits = jnp.zeros((*x.shape, 4), dtype=jnp.float32)
        output = (logits, jnp.array(0.5))
        if mutable is None:
            return output
        return output, {
            "scalars": {"model/value": (jnp.array(1.0), jnp.array(2.0))},
            "intermediates": {"features": (jnp.array(3.0),)},
            "plots": {"weights": (jnp.array(4.0),)},
        }


def test_scalar_metadata_flattens_scopes_and_keeps_last_emission() -> None:
    metadata = scalar_metadata(
        {
            "loss/root": (jnp.array(1.0),),
            "block": {"loss/aux": (jnp.array(2.0), jnp.array(3.0))},
        }
    )

    assert metadata == {
        "loss/root": jnp.array(1.0),
        "block/loss/aux": jnp.array(3.0),
    }


def test_scalar_metadata_rejects_non_scalar_values() -> None:
    with pytest.raises(ValueError, match="must be rank zero"):
        scalar_metadata({"bad": (jnp.ones((2,)),)})


def test_scalar_metadata_rejects_reserved_and_ambiguous_names() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        scalar_metadata({"plots": (jnp.array(1.0),)})

    with pytest.raises(ValueError, match="is ambiguous"):
        scalar_metadata(
            {
                "block/loss": (jnp.array(1.0),),
                "block": {"loss": (jnp.array(2.0),)},
            }
        )


def test_base_forward_returns_flat_scalar_metadata() -> None:
    apply = RecordingApply()
    state: Any = SimpleNamespace(apply_fn=apply)
    batch: Any = {
        "x": jnp.ones((1, 2), dtype=jnp.int32),
        "y": jnp.ones((1, 2), dtype=jnp.int32),
        "padding_mask": jnp.ones((1, 2), dtype=jnp.bool_),
    }

    _, _, meta = BaseTrainer.forward(state, {}, batch)

    assert apply.mutable_calls == [["scalars"]]
    assert meta == {"model/value": jnp.array(2.0)}


def test_ppo_collects_diagnostics_from_policy_only() -> None:
    apply = RecordingApply()
    state: Any = SimpleNamespace(
        apply_fn=apply,
        base={},
        beta=jnp.array(0.1),
        clip_eps=jnp.array(0.2),
    )
    batch: Any = {
        "x": jnp.ones((1, 2), dtype=jnp.int32),
        "y": jnp.ones((1, 2), dtype=jnp.int32),
        "padding_mask": jnp.ones((1, 2), dtype=jnp.bool_),
        "action_mask": jnp.ones((1, 2), dtype=jnp.bool_),
        "old_log_probs": jnp.zeros((1, 2), dtype=jnp.float32),
        "per_token_rewards": jnp.ones((1, 2), dtype=jnp.float32),
    }

    _, _, meta = PPOTrainer.forward(state, {}, batch, intermediates=True)

    assert apply.mutable_calls == [
        ["scalars", "intermediates", "plots"],
        None,
    ]
    assert meta["model/value"] == jnp.array(2.0)
    assert meta["intermediates"] == {"features": (jnp.array(3.0),)}
    assert meta["plots"] == {"weights": (jnp.array(4.0),)}
